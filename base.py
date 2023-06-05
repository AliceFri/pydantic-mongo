from enum import Enum
from typing import List, Dict, Any, Tuple, Union
import datetime
import bson
import pydantic
import mongoengine
from pymongo.collection import Collection
import typing
import inspect

from .db import get_collection
from .const import *
from .f import _BaseModel, cache


class ObjectId(bson.ObjectId):
    @classmethod
    def __get_validators__(cls):  # type: ignore
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema: Dict) -> None:
        field_schema.update(
            examples=["5f85f36d6dfecacc68428a46", "ffffffffffffffffffffffff"],
            example="5f85f36d6dfecacc68428a46",
            type="string",
        )

    @classmethod
    def validate(cls, v: Any) -> bson.ObjectId:
        if isinstance(v, (bson.ObjectId, cls)):
            return v
        if isinstance(v, str) and bson.ObjectId.is_valid(v):
            return bson.ObjectId(v)
        raise TypeError("invalid ObjectId specified")


def get_find_pipeline(sort: None, limit: None, skip: None, **query) -> List[Dict]:
    pipeline = BASE_PIPELINE.copy()
    if query:
        pipeline.append({"$match": query})
    if sort:
        pipeline.append({"$sort": sort})
    if skip and skip > 0:
        pipeline.append({"$skip": skip})
    if limit and limit > 0:
        pipeline.append({"$limit": limit})
    return pipeline


def is_ref_model(cls: type) -> bool:
    try:
        return issubclass(cls, BaseModel)
    except Exception:
        return False


def is_lst_ref_model(cls: typing) -> bool:
    try:
        return cls._name == 'List' and issubclass(cls.__args__[0], BaseModel)
    except Exception:
        return False


class BaseModelMetaclass(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **attrs):
        # 添加主键id, 暫不支持自定义
        if (
            namespace.get('__annotations__')
            and 'id' not in namespace['__annotations__']
        ):
            namespace['__annotations__']['id'] = ObjectId
            namespace['id'] = None

        cls = super().__new__(mcs, name, bases, namespace, **attrs)
        # 保存引用信息
        cls.__ref_fields__ = {}
        cls.__lst_ref_fields__ = {}
        for name, annotation_cls in cls.__annotations__.items():
            if is_ref_model(annotation_cls):
                cls.__ref_fields__[name] = annotation_cls
            elif is_lst_ref_model(annotation_cls):
                cls.__lst_ref_fields__[name] = annotation_cls.__args__[0]

        # 创建索引
        for index in getattr(cls.Config, '_indexs', []):
            if len(index) >= 2:
                cls.get_collection().create_index(index[0], **index[1])
            else:
                cls.get_collection().create_index(index[0])
        # 保存delete_rule
        for ref_cls, field, rule in getattr(cls.Config, '_delete_rule', []):
            ref_cls.register_delete_rule(cls, field, rule)

        return cls


class BaseModel(_BaseModel, metaclass=BaseModelMetaclass):
    @classmethod
    def get_collection_name(cls) -> str:
        if '_collection_name' not in cls.__dict__:
            cls._collection_name = (
                "".join("_%s" % c if c.isupper() else c for c in cls.__name__)
                .strip("_")
                .lower()
            )
        return cls._collection_name

    @classmethod
    def _get_collection_name(cls) -> str:
        """
        for compatible mongoengine, generate pipeline
        """
        return cls.get_collection_name()

    @classmethod
    def get_collection(cls) -> Collection:
        return get_collection(cls.get_collection_name())

    @classmethod
    def get_ref(cls) -> Dict[str, 'BaseModel']:
        return getattr(cls, '__ref_fields__', {})

    @classmethod
    def get_lst_ref(cls) -> Dict[str, 'BaseModel']:
        return getattr(cls, '__lst_ref_fields__', {})

    @classmethod
    def get_delete_rule(cls) -> Dict[Tuple['BaseModel', str], int]:
        return getattr(cls, '_delete_rule', {})

    @classmethod
    def aggregate(cls, pipeline: List[dict]) -> List[dict]:
        page = list(cls.get_collection().aggregate(pipeline, allowDiskUse=True, maxTimeMS=10000))
        # page = list(cls.get_collection().aggregate(pipeline, allowDiskUse=True, maxTimeMS=100))
        return page

    @classmethod
    def count(cls, pipeline: List[Dict]) -> int:
        pipeline = pipeline + [{"$count": "total"}]  # 不用append
        data = cls.aggregate(pipeline)
        if not data:
            return 0
        return data[0]["total"]

    @classmethod
    def objects(cls, **query) -> List[Dict[str, Any]]:
        if 'id' in query:
            query['_id'] = query.pop('id')
        return list(cls.get_collection().find(query))

    @classmethod
    def insert_one(cls, **doc):
        doc = cls.mdict(doc)
        return cls.get_collection().insert_one(doc)

    @classmethod
    def insert_many(cls, doc_list: List[Dict[str, Any]]):
        doc = [cls.mdict(d) for d in doc_list]
        if not doc:
            return []
        return cls.get_collection().insert_many(doc)

    @classmethod
    def create_index(cls, keys, **kwargs):
        cls.get_collection().create_index(keys, **kwargs)

    @classmethod
    def parse_obj(cls, doc: Dict[str, Any]) -> Union['BaseModel', None]:
        if not doc:
            return None
        d = {}
        for k, v in doc.items():
            if v is None:
                continue
            elif k in cls.get_ref():
                d[k] = cls.get_ref()[k].parse_obj(v)
            elif k in cls.get_lst_ref():
                d[k] = [cls.get_lst_ref()[k].parse_obj(i) for i in v]
            else:
                d[k] = v
        return cls(**d)

    @classmethod
    def _cascade_find_pipeline(cls) -> List[dict]:
        def _cascade_lookup(ref_cls, name, get_first=True, list_lookup=False):
            match = {"$expr": {"$eq": [f"$${name}", "$_id"]}}
            if list_lookup:
                match = {
                    "$and": [
                        {"$expr": {"$eq": [{'$type': f"$${name}"}, "array"]}},
                        {"$expr": {"$in": ["$_id", f"$${name}"]}},
                    ],
                }

            p = [
                {
                    "$lookup": {
                        "from": ref_cls._get_collection_name(),
                        # "localField": name,
                        # "foreignField": "_id",
                        "as": name,
                        "let": {name: f"${name}"},
                        "pipeline": [
                            {"$match": match},
                            *BASE_PIPELINE,
                            *ref_cls._cascade_find_pipeline(),
                        ],
                    }
                }
            ]
            if get_first:
                p.append(
                    {
                        '$set': {
                            name: {'$arrayElemAt': [f'${name}', 0]},
                        },
                    }
                )
            return p

        pipeline = []
        for name, ref_cls in cls.get_ref().items():
            pipeline.extend(_cascade_lookup(ref_cls, name))
        for name, ref_cls in cls.get_lst_ref().items():
            pipeline.extend(
                _cascade_lookup(ref_cls, name, get_first=False, list_lookup=True)
            )

        return pipeline

    @classmethod
    def find_one(cls, sort=None, **query) -> Union['BaseModel', None]:
        """
        查询一条记录
        """

        result = cls.find(sort=sort, limit=1, **query)
        if not result:
            return None
        return result[0]

    @classmethod
    def no_ref_find_one(
        cls,
        sort=None,
        skip=None,
        project=None,
        extra_pipeline=None,
        **query,
    ) -> Union['BaseModel', None, Dict[str, Any]]:
        result = cls.no_ref_find(
            sort=sort,
            limit=1,
            skip=skip,
            project=project,
            extra_pipeline=extra_pipeline,
            **query,
        )
        if not result:
            return None
        return result[0]

    @classmethod
    def no_ref_find(
        cls,
        sort=None,
        limit=None,
        skip=None,
        project=None,
        extra_pipeline=None,
        **query,
    ) -> List['BaseModel']:
        pipeline = get_find_pipeline(sort, limit, skip, **query)
        if project:
            pipeline.append({"$project": project})
        if extra_pipeline:
            pipeline.extend(extra_pipeline)
        return cls.aggregate(pipeline)

    @classmethod
    def find(
        cls,
        sort=None,
        limit=None,
        skip=None,
        after_query=None,
        extra_pipeline=None,
        **query,
    ) -> List['BaseModel']:
        pipeline = get_find_pipeline(sort, limit, skip, **query)
        if extra_pipeline:
            pipeline.extend(extra_pipeline)
        pipeline.extend(cls._cascade_find_pipeline())
        if after_query:
            pipeline.append({"$match": after_query})
        return [cls.parse_obj(doc) for doc in cls.aggregate(pipeline)]

    @classmethod
    def page_find(
        cls,
        sort=None,
        limit=None,
        skip=None,
        after_query=None,
        extra_pipeline=None,
        **query,
    ) -> Tuple[int, List['BaseModel']]:
        count = cls.get_collection().count_documents(query)
        data = cls.find(sort, limit, skip, after_query, extra_pipeline, **query)
        return count, data

    @classmethod
    def delete_many(cls, **query):
        """
        支持 批量 删除
        """
        objs = cls.no_ref_find(project={'id': 1}, **query)
        ids = [o['id'] for o in objs]
        cls.get_collection().delete_many({'_id': {'$in': ids}})
        # ref_delete
        for (ref_cls, field), rule in cls.get_delete_rule().items():
            ref_query = {field: {'$in': ids}}
            if rule == DELETE_RULE_SET_NULL:
                ref_cls.get_collection().update_many(
                    ref_query,
                    {'$set': {field: None}},
                )
            elif rule == DELETE_RULE_CASCADE:
                ref_cls.delete_many(**ref_query)
            elif rule == DELETE_RULE_PULL:
                ref_cls.get_collection().update_many(
                    {'$pull': ref_query},
                )

    @classmethod
    def register_delete_rule(cls, ref_cls, field, rule):
        delete_rule = getattr(cls, '_delete_rule', {})
        delete_rule[(ref_cls, field)] = rule
        cls._delete_rule = delete_rule

    @classmethod
    def mdict(cls, input_dict):
        """
        将字典转换为mongodb的字典格式
        """

        def is_exist_model(obj):
            return isinstance(obj, BaseModel) and obj._is_exist()

        d = {}
        for k, v in input_dict.items():
            if k is None:
                continue
            if k in cls.get_ref() and is_exist_model(v):
                # 引用类型存在, 转成OID，如果传进来的就是OID，则不做处理
                d[k] = v.id
            elif k in cls.get_lst_ref() and isinstance(v, list):
                d[k] = [i.id if is_exist_model(i) else i for i in v]
            elif isinstance(v, list):
                if len(v) == 0:
                    d[k] = v
                elif isinstance(v[0], Enum):
                    d[k] = [x.value for x in v]
                else:
                    d[k] = v
            elif isinstance(v, Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    def _is_exist(self) -> bool:
        """
        检查数据库中是否存在ID相同的记录
        """
        if self.id and self.get_collection().find_one({"_id": self.id}):
            return True
        return False

    def doc(self) -> Dict[str, Any]:
        """
        转换为mongodb的字典格式
        """

        return self.mdict(self.__dict__)

    def save(self):
        if self._is_exist():
            self.update()
            return
        doc = self.doc()
        doc.pop('id', None)
        res = self.get_collection().insert_one(doc)
        self.id = res.inserted_id

    def delete(self):
        if not self._is_exist():
            raise Exception("is not exist")
        self.delete_many(id=self.id)

    def update(self, **update_dict):  # 传入参数暂不支持更新引用类型
        if not self._is_exist():
            raise Exception("is not exist")

        if not update_dict:
            update_dict = self.doc()
        else:
            update_dict = self.mdict(update_dict)
        update_dict.pop('id', None)
        self.get_collection().update_one({'_id': self.id}, {'$set': update_dict})


@cache
def create_document_model(
    m: mongoengine.fields.BaseDocument,
    include: list[str] = None,
    exclude: list[str] = None,
    optional: list[str] = None,
) -> _BaseModel:
    """
    mongodb model to schema
    mongoengine定义的模型转换为fastapi的模型

    fastapi通过遍历类中的 __annotations__ 来确定每个字段的类型
    通过遍历全部类型为Field的类成员来确定字段的其他信息

    include 和 exclude 可以同时设置
    但是必须确保同一个模型中不能同时存在
    比如可以设置 include = ['id'], exclude = ['field.id']
    但不可以设置 include = ['id'], exclude = ['another_field']

    支持 Optional
    """
    fields_map: dict[str, mongoengine.fields.BaseField] = m._fields
    fields_ordered: list[str] = m._fields_ordered

    def _parse_clude(_clude: list[str]):
        main = set()
        sub = {}
        if _clude:
            for _n in _clude:
                _n = _n.split('.')
                if len(_n) > 1:
                    sub.setdefault(_n[0], []).append('.'.join(_n[1:]))
                else:
                    main.add(_n[0])
        return main, sub

    def _is_required(f: mongoengine.fields.BaseField):
        if optional and f.name in optional:
            return False
        return f.required or f.name == 'id'

    def _parse_name():
        class_name = f'{m.__name__}_create_document_model'
        if include:
            class_name += f' include {include}'
        if exclude:
            class_name += f' exclude {exclude}'
        if optional:
            class_name += f' optional {optional}'
        return class_name

    main_include, sub_include = _parse_clude(include)
    main_exclude, sub_exclude = _parse_clude(exclude)
    main_optional, sub_optional = _parse_clude(optional)

    if main_include and main_exclude:
        raise Exception('不能同时设置 include 和 exclude')

    new_fields = {}
    annotations = {}
    ref_fields = {}
    list_ref_fields = {}
    for n in fields_ordered:
        if main_include:
            if n not in main_include and n not in sub_include:
                continue
        elif main_exclude:
            if n in main_exclude and m not in sub_exclude:
                continue

        f: mongoengine.fields.BaseField = fields_map[n]
        if type(f) == mongoengine.StringField:
            annotations[n] = str
        elif type(f) == mongoengine.IntField:
            annotations[n] = int
        elif type(f) == mongoengine.BooleanField:
            annotations[n] = bool
        elif type(f) == mongoengine.DateField:
            annotations[n] = datetime.date
        elif type(f) == mongoengine.DateTimeField:
            annotations[n] = datetime.datetime
        elif type(f) == mongoengine.ObjectIdField:
            annotations[n] = ObjectId
        elif type(f) == mongoengine.ListField:
            if type(f.field) == mongoengine.StringField:
                annotations[n] = list[str]
            elif type(f.field) == mongoengine.ReferenceField:
                _kwargs = {}
                if n in sub_include:
                    _kwargs['include'] = sub_include[n]
                if n in sub_exclude:
                    _kwargs['exclude'] = sub_exclude[n]
                if n in sub_optional:
                    _kwargs['optional'] = sub_optional[n]
                annotations[n] = list[
                    create_document_model(f.field.document_type, **_kwargs)
                ]
                list_ref_fields[n] = f.field.document_type
            elif type(f.field) == mongoengine.EmbeddedDocumentField:
                annotations[n] = list[create_document_model(f.field.document_type)]
            elif type(f.field) == mongoengine.DateField:
                annotations[n] = list[datetime.date]
            elif type(f.field) == mongoengine.EnumField:
                # 暂时无用 因为mongoengione目前EnumField有bug 无法通过document校验
                annotations[n] = list[f.field._enum_cls]
            elif type(f.field) == mongoengine.DictField:
                annotations[n] = list[dict]
            elif type(f.field) == mongoengine.IntField:
                annotations[n] = list[int]
            else:
                raise Exception(f'不支持的ListField类型 {type(f.field)} {n}')
        elif type(f) == mongoengine.ReferenceField:

            _kwargs = {}
            if n in sub_include:
                _kwargs['include'] = sub_include[n]
            if n in sub_exclude:
                _kwargs['exclude'] = sub_exclude[n]

            annotations[n] = create_document_model(f.document_type, **_kwargs)
            ref_fields[n] = f.document_type
        elif type(f) == mongoengine.EnumField:
            annotations[n] = type(f.choices[0])
        else:
            raise Exception(f'无法解析的字段 {n} {type(f)}')

        kwargs = {}
        title = getattr(f, 'title', None)
        if title:
            kwargs['title'] = title

        if inspect.isfunction(f.default):
            default = f.default()
        else:
            default = f.default
        new_fields[n] = pydantic.Field(
            Ellipsis if _is_required(f) else default,
            **kwargs,
        )

    new_fields['__annotations__'] = annotations

    new_fields['__F_ref_fields__'] = ref_fields
    new_fields['__F_list_ref_fields__'] = list_ref_fields

    class_name = m.__name__
    if include:
        class_name += f' include {include}'
    if exclude:
        class_name += f' exclude {exclude}'

    return pydantic.create_model(
        _parse_name(),
        **new_fields,
        __base__=_BaseModel,
        __ref_fields__=getattr(m, '__ref_fields__', {}),
    )


@cache
def create_document_page_model(
    m: mongoengine.fields.BaseDocument,
    include: list[str] = None,
    exclude: list[str] = None,
) -> _BaseModel:
    """
    类似于create_document_model, 加入total字段, 用于分页
    """
    m = create_document_model(m, include=include, exclude=exclude)
    return create_page_model(m)


def create_page_model(m: pydantic.BaseModel) -> _BaseModel:
    new_fields = {
        'total': (int, ...),
        'data': (list[m], []),
    }

    return pydantic.create_model(
        f'page_response{m.__name__}',
        **new_fields,
        __base__=_BaseModel,
    )
