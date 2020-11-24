# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

"""Stanford 3D Objects for Disentangling (S3O4D) dataset."""

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = r"""
# Stanford 3D Objects for Disentangling (S3O4D)

![Stanford 3D Objects for Disentangling](s3o4d.png)

This folder contains the dataset first described in the "Stanford 3D Objects"
section of the paper [Disentangling by Subspace Diffusion](https://arxiv.org/abs/2006.12982).
The data consists of 100,000 renderings each of the Bunny and Dragon objects
from the [Stanford 3D Scanning Repository](http://graphics.stanford.edu/data/3Dscanrep/).
More objects may be added in the future, but only the Bunny and Dragon are used
in the paper. Each object is rendered with a uniformly sampled illumination from
a point on the 2-sphere, and a uniformly sampled 3D rotation. The true latent
states are provided as NumPy arrays along with the images. The lighting is given
as a 3-vector with unit norm, while the rotation is provided both as a
quaternion and a 3x3 orthogonal matrix.

## Why another dataset?

There are many similarities between S3O4D and existing ML benchmark datasets
like [NORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0/),
[3D Chairs](https://github.com/mathieuaubry/seeing3Dchairs),
[3D Shapes](https://github.com/deepmind/3d-shapes) and many others, which also
include renderings of a set of objects under different pose and illumination
conditions. However, none of these existing datasets include the *full manifold*
of rotations in 3D - most include only a subset of changes to elevation and
azimuth. S3O4D images are sampled uniformly and independently from the full space
of rotations and illuminations, meaning the dataset contains objects that are
upside down and illuminated from behind or underneath. We believe that this
makes S3O4D uniquely suited for research on generative models where the latent
space has non-trivial topology, as well as for general manifold learning
methods where the curvature of the manifold is important.

## Usage

The data can be loaded from [TensorFlow Datasets](https://www.tensorflow.org/datasets)
using the standard interface, or it can be manually downloaded and extracted as
follows. To load the data for a given object, unzip `images.zip` into a folder
called `images` in the same directory as `latents.npz`, and from inside that
directory run:

```
import numpy as np
from PIL import Image

with open('latents.npz', 'r') as f:
  data = np.load(f)
  illumination = data['illumination']  # lighting source position, a 3-vector
  pose_quat = data['pose_quat']  # object pose (3D rotation as a quaternion)
  pose_mat = data['pose_mat']  # object pose (3D rotation as a matrix)

def get_data(i):
  img = np.array(Image.open(f'images/{i:05}.jpg'))
  # Uses the matrix, not quaternion, representation,
  # similarly to the experiments in the paper
  latent = np.concatenate((illumination[i],
  	                       pose_mat[i].reshape(-1)))
  return img, latent

img, latent = get_data(0)
```

If you prefer not to do your own test/train split, the images and latents are
also available pre-split into `train_images.zip`, `test_images.zip`,
`train_latents.npz` and `test_latents.npz`, with an 80/20 train/test split.

## Giving Credit

If you use this data in your work, we ask you to cite this paper:

```
@article{pfau2020disentangling,
  title={Disentangling by Subspace Diffusion},
  author={Pfau, David and Higgins, Irina and Botev, Aleksandar and Racani\`ere,
  S{\'e}bastian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

## Disclaimer

This is not an official Google product.
"""

_CITATION = r"""
@article{pfau2020disentangling,
  title={Disentangling by Subspace Diffusion},
  author={Pfau, David and Higgins, Irina and Botev, Aleksandar and Racani\`ere,
  S{\'e}bastian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
"""


class S3O4D(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for s3o4d dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(256, 256, 3)),
            'label': tfds.features.ClassLabel(names=['bunny', 'dragon']),
            'illumination': tfds.features.Tensor(shape=(3,), dtype=tf.float32),
            'pose_quat': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
            'pose_mat': tfds.features.Tensor(shape=(3, 3), dtype=tf.float32),
        }),
        supervised_keys=None,
        homepage='https://github.com/deepmind/deepmind-research/tree/master/geomancer#stanford-3d-objects-for-disentangling-s3o4d',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    bunny_train_img = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/bunny/train_images.zip')
    bunny_test_img = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/bunny/test_images.zip')
    dragon_train_img = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/dragon/train_images.zip')
    dragon_test_img = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/dragon/test_images.zip')

    bunny_train_latent = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/bunny/train_latents.npz')
    bunny_test_latent = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/bunny/test_latents.npz')
    dragon_train_latent = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/dragon/train_latents.npz')
    dragon_test_latent = dl_manager.download(
        'https://storage.googleapis.com/dm_s3o4d/dragon/test_latents.npz')

    return {
        'bunny_train': self._generate_examples(
            dl_manager, bunny_train_img, bunny_train_latent, 'bunny'),
        'bunny_test': self._generate_examples(
            dl_manager, bunny_test_img, bunny_test_latent, 'bunny'),
        'dragon_train': self._generate_examples(
            dl_manager, dragon_train_img, dragon_train_latent, 'dragon'),
        'dragon_test': self._generate_examples(
            dl_manager, dragon_test_img, dragon_test_latent, 'dragon'),
    }

  def _generate_examples(self, dl_manager: tfds.download.DownloadManager,
                         img_path, latent_path, label):
    """Yields examples."""
    with tf.io.gfile.GFile(latent_path, 'rb') as f:
      latents = dict(np.load(f))
    for key in latents:
      latents[key] = latents[key].astype(np.float32)
    for fname, fobj in dl_manager.iter_archive(img_path):
      idx = int(fname[-9:-4]) % 80000
      yield label + '_' + fname[-9:-4], {
          'image': fobj,
          'label': label,
          'illumination': latents['illumination'][idx],
          'pose_mat': latents['pose_mat'][idx],
          'pose_quat': latents['pose_quat'][idx],
      }
