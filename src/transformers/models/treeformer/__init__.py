# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_treeformer": ["TREEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "TreeformerConfig", "TreeformerOnnxConfig"],
    "tokenization_treeformer": ["TreeformerTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_treeformer_fast"] = ["TreeformerTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_treeformer"] = [
        "TREEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TreeformerForCausalLM",
        "TreeformerForMaskedLM",
        "TreeformerForMultipleChoice",
        "TreeformerForQuestionAnswering",
        "TreeformerForSequenceClassification",
        "TreeformerForTokenClassification",
        "TreeformerModel",
        "TreeformerPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_treeformer"] = [
        "TF_TREEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFTreeformerForCausalLM",
        "TFTreeformerForMaskedLM",
        "TFTreeformerForMultipleChoice",
        "TFTreeformerForQuestionAnswering",
        "TFTreeformerForSequenceClassification",
        "TFTreeformerForTokenClassification",
        "TFTreeformerMainLayer",
        "TFTreeformerModel",
        "TFTreeformerPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_treeformer"] = [
        "FlaxTreeformerForCausalLM",
        "FlaxTreeformerForMaskedLM",
        "FlaxTreeformerForMultipleChoice",
        "FlaxTreeformerForQuestionAnswering",
        "FlaxTreeformerForSequenceClassification",
        "FlaxTreeformerForTokenClassification",
        "FlaxTreeformerModel",
        "FlaxTreeformerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_treeformer import TREEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, TreeformerConfig, TreeformerOnnxConfig
    from .tokenization_treeformer import TreeformerTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_treeformer_fast import TreeformerTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_treeformer import (
            TREEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TreeformerForCausalLM,
            TreeformerForMaskedLM,
            TreeformerForMultipleChoice,
            TreeformerForQuestionAnswering,
            TreeformerForSequenceClassification,
            TreeformerForTokenClassification,
            TreeformerModel,
            TreeformerPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_treeformer import (
            TF_TREEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFTreeformerForCausalLM,
            TFTreeformerForMaskedLM,
            TFTreeformerForMultipleChoice,
            TFTreeformerForQuestionAnswering,
            TFTreeformerForSequenceClassification,
            TFTreeformerForTokenClassification,
            TFTreeformerMainLayer,
            TFTreeformerModel,
            TFTreeformerPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_treeformer import (
            FlaxTreeformerForCausalLM,
            FlaxTreeformerForMaskedLM,
            FlaxTreeformerForMultipleChoice,
            FlaxTreeformerForQuestionAnswering,
            FlaxTreeformerForSequenceClassification,
            FlaxTreeformerForTokenClassification,
            FlaxTreeformerModel,
            FlaxTreeformerPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
