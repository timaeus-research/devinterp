# Copyright 2018 The Lucid Authors. All Rights Reserved.
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
# ==============================================================================

"""Methods for displaying images from Numpy arrays."""

import base64
import logging
import math
from io import BytesIO
from string import Template

import IPython.display
import numpy as np
import PIL.Image

# create logger with module name, e.g. lucid.misc.io.array_to_image
log = logging.getLogger(__name__)


# serialize array

def _normalize_array(array: np.ndarray, domain=(0, 1)) -> np.ndarray:
    array = np.array(array)
    array = np.squeeze(array)
    assert len(array.shape) <= 3
    assert np.issubdtype(array.dtype, np.number)
    assert not np.isnan(array).any()

    low, high = np.min(array), np.max(array)
    if domain is None:
        log.debug("No domain specified, normalizing from measured (~%.2f, ~%.2f)", low, high)
        domain = (low, high)

    if low < domain[0] or high > domain[1]:
        log.info("Clipping domain from (~{:.2f}, ~{:.2f}) to (~{:.2f}, ~{:.2f}).".format(low, high, domain[0], domain[1]))
        array = array.clip(*domain)

    min_value, max_value = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max
    if np.issubdtype(array.dtype, np.inexact):
        offset = domain[0]
        if offset != 0:
            array -= offset
            log.debug("Converting inexact array by subtracting -%.2f.", offset)
        if domain[0] != domain[1]:
            scalar = max_value / (domain[1] - domain[0])
            if scalar != 1:
                array *= scalar
                log.debug("Converting inexact array by scaling by %.2f.", scalar)

    return array.clip(min_value, max_value).astype(np.uint8)


def _serialize_normalized_array(array: np.ndarray, fmt='png', quality=70) -> bytes:
    assert np.issubdtype(array.dtype, np.unsignedinteger)
    assert np.max(array) <= np.iinfo(array.dtype).max
    assert array.shape[-1] > 1

    image = PIL.Image.fromarray(array)
    image_bytes = BytesIO()
    image.save(image_bytes, fmt, quality=quality)
    return image_bytes.getvalue()


def serialize_array(array: np.ndarray, domain=(0, 1), fmt='png', quality=70) -> bytes:
    normalized = _normalize_array(array, domain=domain)
    return _serialize_normalized_array(normalized, fmt=fmt, quality=quality)


JS_ARRAY_TYPES = {'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'float32', 'float64'}


def array_to_jsbuffer(array: np.ndarray) -> str:
    if array.ndim != 1:
        raise TypeError('Only 1d arrays can be converted JS TypedArray.')
    if array.dtype.name not in JS_ARRAY_TYPES:
        raise TypeError('Array dtype not supported by JS TypedArray.')

    js_type_name = array.dtype.name.capitalize() + 'Array'
    data_base64 = base64.b64encode(array.tobytes()).decode('ascii')
    code = f"""
    (function() {{
      const data = atob("{data_base64}");
      const buf = new Uint8Array(data.length);
      for (var i=0; i<data.length; ++i) {{
        buf[i] = data.charCodeAt(i);
      }}
      var array_type = {js_type_name};
      if (array_type == Uint8Array) {{
        return buf;
      }}
      return new array_type(buf.buffer);
    }})()
    """
    return code


# collapse channels

def hue_to_rgb(ang: float, warp: bool = True) -> np.ndarray:
    """Produce an RGB unit vector corresponding to a hue of a given angle."""
    ang = ang - 360 * (ang // 360)
    colors = np.asarray([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
    ])
    colors /= np.linalg.norm(colors, axis=1, keepdims=True)
    R = 360 / len(colors)
    n = math.floor(ang / R)
    D = (ang - n * R) / R

    if warp:
        adj = lambda x: math.sin(x * math.pi / 2)
        if n % 2 == 0:
            D = adj(D)
        else:
            D = 1 - adj(1 - D)

    v = (1 - D) * colors[n] + D * colors[(n + 1) % len(colors)]
    return v / np.linalg.norm(v)


def sparse_channels_to_rgb(X: np.ndarray) -> np.ndarray:
    assert (X >= 0).all()

    K = X.shape[-1]
    rgb = 0

    for i in range(K):
        ang = 360 * i / K
        color = hue_to_rgb(ang)
        color = color[tuple(None for _ in range(len(X.shape) - 1))]
        rgb += X[..., i, None] * color

    rgb += np.ones(X.shape[:-1])[..., None] * (X.sum(-1) - X.max(-1))[..., None]
    rgb /= 1e-4 + np.linalg.norm(rgb, axis=-1, keepdims=True)
    rgb *= np.linalg.norm(X, axis=-1, keepdims=True)

    return rgb


def collapse_channels(X: np.ndarray) -> np.ndarray:
    if (X < 0).any():
        X = np.concatenate([np.maximum(0, X), np.maximum(0, -X)], axis=-1)
    return sparse_channels_to_rgb(X)


# showing


def _display_html(html_str):
  IPython.display.display(IPython.display.HTML(html_str))


def _image_url(array, fmt='png', mode="data", quality=90, domain=None):
  """Create a data URL representing an image from a PIL.Image.

  Args:
    image: a numpy array
    mode: presently only supports "data" for data URL

  Returns:
    URL representing image
  """
  supported_modes = ("data")
  if mode not in supported_modes:
    message = "Unsupported mode '%s', should be one of '%s'."
    raise ValueError(message, mode, supported_modes)

  image_data = serialize_array(array, fmt=fmt, quality=quality, domain=domain)
  base64_byte_string = base64.b64encode(image_data).decode('ascii')
  return "data:image/" + fmt.upper() + ";base64," + base64_byte_string


# public functions

def _image_html(array, w=None, domain=None, fmt='png'):
  url = _image_url(array, domain=domain, fmt=fmt)
  style = "image-rendering: pixelated; image-rendering: crisp-edges;"
  if w is not None:
    style += "width: {w}px;".format(w=w)
  return """<img src="{url}" style="{style}">""".format(**locals())

def image(array, domain=None, w=None, format='png', **kwargs):
  """Display an image.

  Args:
    array: NumPy array representing the image
    fmt: Image format e.g. png, jpeg
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """

  _display_html(
    _image_html(array, w=w, domain=domain, fmt=format)
  )


def images(arrays, labels=None, domain=None, w=None):
  """Display a list of images with optional labels.

  Args:
    arrays: A list of NumPy arrays representing images
    labels: A list of strings to label each image.
      Defaults to show index if None
    domain: Domain of pixel values, inferred from min & max values if None
    w: width of output image, scaled using nearest neighbor interpolation.
      size unchanged if None
  """

  s = '<div style="display: flex; flex-direction: row;">'
  for i, array in enumerate(arrays):
    label = labels[i] if labels is not None else i
    img_html = _image_html(array, w=w, domain=domain)
    s += """<div style="margin-right:10px; margin-top: 4px;">
              {label} <br/>
              {img_html}
            </div>""".format(**locals())
  s += "</div>"
  _display_html(s)


def show(thing, domain=(0, 1), **kwargs):
  """Display a numpy array without having to specify what it represents.

  This module will attempt to infer how to display your tensor based on its
  rank, shape and dtype. rank 4 tensors will be displayed as image grids, rank
  2 and 3 tensors as images.

  For tensors of rank 3 or 4, the innermost dimension is interpreted as channel.
  Depending on the size of that dimension, different types of images will be
  generated:

    shp[-1]
      = 1  --  Black and white image.
      = 2  --  See >4
      = 3  --  RGB image.
      = 4  --  RGBA image.
      > 4  --  Collapse into an RGB image.
               If all positive: each dimension gets an evenly spaced hue.
               If pos and neg: each dimension gets two hues
                  (180 degrees apart) for positive and negative.

  Common optional arguments:

    domain: range values can be between, for displaying normal images
      None  = infer domain with heuristics
      (a,b) = clip values to be between a (min) and b (max).

    w: width of displayed images
      None  = display 1 pixel per value
      int   = display n pixels per value (often used for small images)

    labels: if displaying multiple objects, label for each object.
      None  = label with index
      []    = no labels
      [...] = label with corresponding list item

  """
  def collapse_if_needed(arr):
    K = arr.shape[-1]
    if K not in [1,3,4]:
      log.debug("Collapsing %s channels into 3 RGB channels." % K)
      return collapse_channels(arr)
    else:
      return arr


  if isinstance(thing, np.ndarray):
    rank = len(thing.shape)

    if rank in [3,4]:
      thing = collapse_if_needed(thing)

    if rank == 4:
      log.debug("Show is assuming rank 4 tensor to be a list of images.")
      images(thing, domain=domain, **kwargs)
    elif rank in (2, 3):
      log.debug("Show is assuming rank 2 or 3 tensor to be an image.")
      image(thing, domain=domain, **kwargs)
    else:
      log.warning("Show only supports numpy arrays of rank 2-4. Using repr().")
      print(repr(thing))
  elif isinstance(thing, (list, tuple)):
    log.debug("Show is assuming list or tuple to be a collection of images.")

    if isinstance(thing[0], np.ndarray) and len(thing[0].shape) == 3:
      thing = [collapse_if_needed(t) for t in thing]

    images(thing, domain=domain, **kwargs)
  else:
    log.warning("Show only supports numpy arrays so far. Using repr().")
    print(repr(thing))

