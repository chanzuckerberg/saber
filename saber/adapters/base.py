from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Literal, Optional, Union, Iterator, Tuple, Any, Dict, List
import numpy as np


class SAM2AdapterConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_type: Literal["sam2"] = "sam2"
    cfg: str = Field("small", description="tiny / small / base / large")
    checkpoint: Optional[str] = None
    num_maskmem: int = 2
    light_modality: bool = False
    amg_cfg: Optional[Any] = None  # cfgAMG instance; None → cfgAMG() defaults
    min_mask_area: int = 50
    classifier: Optional[Any] = None  # Predictor; auto-fills cfg + amg_cfg when provided

    @model_validator(mode='after')
    def _derive_from_classifier(self) -> 'SAM2AdapterConfig':
        if self.classifier is not None and self.amg_cfg is None:
            from saber.adapters.sam2.amg import cfgAMG
            amg_params = self.classifier.config['amg_params']
            self.cfg = amg_params.get('sam2_cfg', self.cfg)
            self.amg_cfg = cfgAMG(**amg_params)
        return self

    @field_validator("cfg")
    @classmethod
    def _check_cfg(cls, v):
        if v not in {"tiny", "small", "base", "large"}:
            raise ValueError(f"cfg must be one of tiny/small/base/large, got '{v}'")
        return v


class SAM3AdapterConfig(BaseModel):
    model_type: Literal["sam3"] = "sam3"
    checkpoint_path: Optional[str] = None
    load_from_HF: bool = True
    light_modality: bool = False
    text_prompt: Optional[str] = None
    min_mask_area: int = 50


AdapterConfig = Union[SAM2AdapterConfig, SAM3AdapterConfig]


class BaseAdapter(ABC):
    """Common interface every tomogram adapter must implement."""

    # Populated by segment_volume() — {frame_idx: {obj_id: {"presence_score": float, ...}}}
    frame_metrics: Dict[int, Dict[int, Dict[str, Any]]]

    @abstractmethod
    def segment_image_2d(
        self,
        image: np.ndarray,
        text_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        2D detection/segmentation.
        Returns list of dicts with at minimum: {'segmentation': (H,W) bool, 'area': int}
        Matches the format returned by SAM2 AMG so callers are model-agnostic.
        """
        ...

    @abstractmethod
    def set_volume(self, tomogram: np.ndarray, offload_video_to_cpu: bool = False) -> None: ...

    @abstractmethod
    def add_new_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray,
                     inference_state=None) -> Tuple: ...

    @abstractmethod
    def add_new_points_or_box(self, frame_idx: int, obj_id: int,
                               inference_state=None, **kwargs) -> Tuple: ...

    @abstractmethod
    def propagate_in_video(self, start_frame_idx, max_frame_num_to_track=None,
                            reverse=False, inference_state=None) -> Iterator: ...

    @abstractmethod
    def segment_volume(self, start_frame_idx: int, masks=None,
                       vol_shape=None, max_frame_num_to_track=None,
                       min_presence_score: float = 0.5,
                       inference_state=None) -> np.ndarray: ...

    @abstractmethod
    def reset_state(self, inference_state=None) -> None: ...


def get_adapter(config: AdapterConfig, device: str = "cuda") -> BaseAdapter:
    if config.model_type == "sam2":
        from saber.adapters.sam2 import SAM2Adapter
        return SAM2Adapter(config, device)
    from saber.adapters.sam3 import SAM3Adapter
    return SAM3Adapter(config, device)
