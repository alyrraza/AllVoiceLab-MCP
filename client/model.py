from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Voice:
    voice_id: str = ""
    name: str = ""
    description: str = ""
    files: List[str] = field(default_factory=list)
    is_legacy: bool = False
    is_favor: bool = False
    labels: Dict[str, str] = field(default_factory=dict)
    voice_settings: Dict[str, int] = field(default_factory=dict)
    icon_url: str = ""
    status: int = 0
    fail_code: int = 0
    is_pvc: bool = False
    is_disabled: bool = False
    created_time_sec: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'Voice':
        return cls(
            voice_id=data.get('voice_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            files=data.get('files', []),
            is_legacy=data.get('is_legacy', False),
            is_favor=data.get('is_favor', False),
            labels=data.get('labels', {}),
            voice_settings=data.get('voice_settings', {}),
            icon_url=data.get('icon_url', ''),
            status=data.get('status', 0),
            fail_code=data.get('fail_code', 0),
            is_pvc=data.get('is_pvc', False),
            is_disabled=data.get('is_disabled', False),
            created_time_sec=data.get('created_time_sec', 0)
        )


@dataclass
class GetAllVoicesResponse:
    voices: List[Voice] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GetAllVoicesResponse':
        return cls(
            voices=[Voice.from_dict(voice_data) for voice_data in data.get('voices', [])]
        )


@dataclass
class GetSupportedVoiceModelResponse:
    models: List['VoiceModel']

    @classmethod
    def from_dict(cls, data):
        return cls(
            models=[VoiceModel.from_dict(model) for model in data.get('models', [])]
        )


@dataclass
class VoiceModel:
    model_id: str
    name: str
    can_do_text_to_speech: bool
    can_do_voice_conversion: bool
    description: str
    languages: List['VoiceModelLanguage']

    @classmethod
    def from_dict(cls, data):
        return cls(
            model_id=data.get('model_id', ''),
            name=data.get('name', ''),
            can_do_text_to_speech=data.get('can_do_text_to_speech', False),
            can_do_voice_conversion=data.get('can_do_voice_conversion', False),
            description=data.get('description', ''),
            languages=[VoiceModelLanguage.from_dict(lang) for lang in data.get('languages', [])]
        )


@dataclass
class VoiceModelLanguage:
    language_id: str
    name: str

    @classmethod
    def from_dict(cls, data):
        return cls(
            language_id=data.get('language_id', ''),
            name=data.get('name', '')
        )


@dataclass
class DubbingInfoResponse:
    dubbing_id: str = ""
    name: str = ""
    status: str = ""
    target_languages: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DubbingInfoResponse':
        return cls(
            dubbing_id=data.get('dubbing_id', ''),
            name=data.get('name', ''),
            status=data.get('status', ''),
            target_languages=data.get('target_languages', [])
        )


@dataclass
class RemovalInfoResponse:
    project_id: str = ""
    name: str = ""
    status: str = ""
    language_code: str = ""
    removal_result: str = ""
    extraction_result: str = ""
    output_url: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RemovalInfoResponse':
        return cls(
            project_id=data.get('project_id', ''),
            name=data.get('name', ''),
            status=data.get('status', ''),
            language_code=data.get('language_code', ''),
            removal_result=data.get('removal_result', ''),
            extraction_result=data.get('extraction_result', ''),
            output_url=data.get('output_url', '')
        )


@dataclass
class ExtractionInfoResponse:
    project_id: str = ""
    name: str = ""
    status: str = ""
    language_code: str = ""
    result: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionInfoResponse':
        return cls(
            project_id=data.get('project_id', ''),
            name=data.get('name', ''),
            status=data.get('status', ''),
            language_code=data.get('language_code', ''),
            result=data.get('result', '')
        )


@dataclass
class TextTranslationResultResponse:
    project_id: str = ""
    name: str = ""
    status: str = ""
    source_lang: str = ""
    target_lang: str = ""
    result: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> 'TextTranslationResultResponse':
        return cls(
            project_id=data.get('project_id', ''),
            name=data.get('name', ''),
            status=data.get('status', ''),
            source_lang=data.get('source_lang', ''),
            target_lang=data.get('target_lang', ''),
            result=data.get('result', '')
        )
