from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ACCEPT_WITH_WARNINGS = "ACCEPT_WITH_WARNINGS"
    AUTO_FIXED_ACCEPT = "AUTO_FIXED_ACCEPT"
    NEED_REUPLOAD = "NEED_REUPLOAD"


class ReasonCode(str, Enum):
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    PDF_ENCRYPTED = "PDF_ENCRYPTED"
    PAGE_BLANK = "PAGE_BLANK"
    PAGE_TOO_LOW_RESOLUTION = "PAGE_TOO_LOW_RESOLUTION"
    SEVERE_BLUR = "SEVERE_BLUR"
    SEVERE_EXPOSURE_ABNORMAL = "SEVERE_EXPOSURE_ABNORMAL"
    NO_EFFECTIVE_TEXT_DETECTED = "NO_EFFECTIVE_TEXT_DETECTED"
    TEXT_TOO_SMALL_RISK = "TEXT_TOO_SMALL_RISK"
    LOCAL_BLUR_RISK = "LOCAL_BLUR_RISK"
    BORDER_CROP_RISK = "BORDER_CROP_RISK"
    OCR_PROBE_LOW_CONFIDENCE = "OCR_PROBE_LOW_CONFIDENCE"
    LOW_CONTRAST_RISK = "LOW_CONTRAST_RISK"
    COMPRESSION_ARTIFACT_RISK = "COMPRESSION_ARTIFACT_RISK"
    ORIENTATION_CORRECTED = "ORIENTATION_CORRECTED"
    ORIENTATION_DETECTION_UNAVAILABLE = "ORIENTATION_DETECTION_UNAVAILABLE"
    HAND_OCCLUSION_RISK = "HAND_OCCLUSION_RISK"
    TEXT_OCCLUSION_RISK = "TEXT_OCCLUSION_RISK"


USER_MESSAGES = {
    ReasonCode.UNSUPPORTED_FILE_TYPE: "文件格式暂不支持，请上传图片或 PDF。",
    ReasonCode.FILE_CORRUPTED: "文件可能已损坏，请重新上传清晰完整的文件。",
    ReasonCode.IMAGE_DECODE_FAILED: "图片无法解码，请重新上传有效图片。",
    ReasonCode.PDF_ENCRYPTED: "PDF 无法处理，请上传未加密文件。",
    ReasonCode.PAGE_BLANK: "页面未检测到有效内容，请重新上传。",
    ReasonCode.PAGE_TOO_LOW_RESOLUTION: "页面分辨率过低，可能无法支持 OCR 识别。",
    ReasonCode.SEVERE_BLUR: "页面严重模糊，请重新拍摄或扫描。",
    ReasonCode.SEVERE_EXPOSURE_ABNORMAL: "页面曝光异常，请重新拍摄或扫描。",
    ReasonCode.NO_EFFECTIVE_TEXT_DETECTED: "页面未检测到足够文字内容。",
    ReasonCode.TEXT_TOO_SMALL_RISK: "页面文字偏小，可能影响 OCR 识别。",
    ReasonCode.LOCAL_BLUR_RISK: "页面存在局部模糊风险。",
    ReasonCode.BORDER_CROP_RISK: "页面文字疑似贴近边界，可能存在裁切风险。",
    ReasonCode.OCR_PROBE_LOW_CONFIDENCE: "OCR 抽样探测置信度较低，请重新上传更清晰文件。",
    ReasonCode.LOW_CONTRAST_RISK: "页面对比度偏低，可能影响 OCR 识别。",
    ReasonCode.COMPRESSION_ARTIFACT_RISK: "页面疑似存在压缩失真风险。",
    ReasonCode.ORIENTATION_CORRECTED: "页面方向已自动修正。",
    ReasonCode.ORIENTATION_DETECTION_UNAVAILABLE: "方向检测暂不可用，已跳过方向修正。",
    ReasonCode.HAND_OCCLUSION_RISK: "页面疑似存在手指或手掌遮挡文字，请确认文字区域完整可见。",
    ReasonCode.TEXT_OCCLUSION_RISK: "页面疑似存在物体遮挡文字，请确认文字区域完整可见。",
}


@dataclass
class AutoFixRecord:
    fixType: str
    before: dict[str, Any]
    after: dict[str, Any]
    recomputed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageResult:
    pageIndex: int
    accepted: bool
    decision: Decision
    ocrReadinessScore: float
    reasonCodes: list[ReasonCode] = field(default_factory=list)
    userMessages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    autoFixes: list[AutoFixRecord] = field(default_factory=list)
    timingsMs: dict[str, float] = field(default_factory=dict)
    textDetectionSummary: dict[str, Any] | None = None
    occlusionSummary: dict[str, Any] | None = None
    ocrProbeSummary: dict[str, Any] | None = None
    ocrProbeExecuted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pageIndex": self.pageIndex,
            "accepted": self.accepted,
            "decision": self.decision.value,
            "ocrReadinessScore": self.ocrReadinessScore,
            "reasonCodes": [code.value for code in self.reasonCodes],
            "userMessages": self.userMessages,
            "metrics": self.metrics,
            "autoFixes": [fix.to_dict() for fix in self.autoFixes],
            "timingsMs": self.timingsMs,
            "textDetectionSummary": self.textDetectionSummary,
            "occlusionSummary": self.occlusionSummary,
            "ocrProbeSummary": self.ocrProbeSummary,
            "ocrProbeExecuted": self.ocrProbeExecuted,
        }


def messages_for(reason_codes: list[ReasonCode]) -> list[str]:
    return [USER_MESSAGES[code] for code in reason_codes if code in USER_MESSAGES]
