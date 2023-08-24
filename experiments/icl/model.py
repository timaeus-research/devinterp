"""
Decode-only transformer architecture retrofit for in-context linear
regression problem.

Notes maybe for the specific problem in mind:

* When it comes time to train, probably I don't need to bother training the
  model to predict the xs?
  * The optimal output for them is zero since they are drawn IID from
    standard normal.
  * There may be some point in checking them since it will regularise the
    rest of the models.
* To be honest this task should probably have been modelled as an
  encoder-decoder problem if we are actually trying to solve it...
  maybe I should experiment with cross attention architectures.
* Relatedly, maybe it's a good idea to provide the model with a hint in
  its positional encoding that distinguishes x and y, and connects adjacent x
  to y...
* Though I guess the point is to study decoded-only transformers without this
  special position information, not to do linear regression... OK fair
  enough. But keep these interventions in mind if we want to elicit the
  behaviour more later.
"""


import torch

from dtransformer import DTransformer


class InContextRegressionTransformer(torch.nn.Module):
    def __init__(
        self,
        task_size,
        max_examples,
        embed_size,
        mlp_size,
        num_heads,
        num_layers,
        device='cpu',
    ):
        super().__init__()
        self.token_sequence_transformer = DTransformer(
            token_size=1 + task_size,    # task_size for x + 1 for y
            max_tokens=2 * max_examples, # one x + one y per example
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_layers=num_layers,
            device=device,
        )
        self.task_size = task_size
        self.max_examples = max_examples
        self.device = device

    
    def forward(self, xs, ys):
        # input validation
        B, K, D = xs.shape
        assert K <= self.max_examples, \
            f"too many examples for model {K} > {self.max_examples}"
        assert D == self.task_size, \
            f"incorrect input size for model {D} != {self.task_size}"
        B_, K_, _1 = ys.shape
        assert B == B_, f"batch size mismatch b/w xs:{B} and ys:{B_}"
        assert K == K_, f"num_examples mismatch b/w xs:{K} and ys:{K_}"
        assert _1 == 1, "ys should be scalars"

        # encode examples as token sequence
        toks = to_token_sequence(xs, ys)
        # run dtransformer to predict next tokens
        toks_pred = self.token_sequence_transformer(toks)
        # decode y predictions from next-token-prediction
        ys_pred = from_predicted_token_sequence(toks_pred)
        return ys_pred


# # # Task encoding / decoding helper functions

    
def to_token_sequence(xs, ys):
    """
    Convert a regression data set into a sequence of vector tokens using a
    concatenating / joint encoding.

    Parameters:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.

    Returns:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded inputs and
        outputs.

    Examples:

    ```
    > xs = [ [1,2,3], [2,3,4], [6,5,4], ]
    > ys = [ 1,       2,       6,       ] # y = x * [ 1, 0, 0, ]
    > to_token_sequence(xs, ys)
      (error! wrong! inputs not batched! and ys not singletons!)

    > xs = [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    > ys = [ [ [1],     [2],     [6],     ] ]
    > to_token_sequence(xs, ys)
      [ [ [ 0, 1, 2, 3, ]       # 0, first x
        , [ 1, 0, 0, 0, ]       # first y, 0s
        , [ 0, 2, 3, 4, ]       # 0, second x
        , [ 2, 0, 0, 0, ]       # ...
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    ```
    """
    B, K, D = xs.shape
    # convert to input-output pairs (of the form [ 0 x1 .. xD  y 0 .. 0 ])
    xys = torch.cat([
        torch.zeros_like(ys),   #   B K 1
        xs,                     # | B K D
        ys,                     # | B K 1
        torch.zeros_like(xs),   # | B K D
    ], dim=-1)                  # -> B K 2D+2
    # convert to token sequences (alternating [ 0 x1 .. xD ], [ y 0 .. 0 ])
    toks = xys.reshape(B, 2*K, D+1)
    return toks


def from_predicted_token_sequence(toks, return_xs=False):
    """
    Convert a sequence of vector next-token-predictions into a regression
    data set by decoding from a joint/concatenating encoding.
    
    Parameters:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded next token
        predictions from inputs and outputs.
    * `return_xs=False : bool`
        whether to decode and return xs, or just ys (default).

    Returns:

    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.
    * `xs : tensor(B, K, D)`
        (only included if `return_xs=True`)
        batch of `B` sequences of `K` input vectors of `D` dims.

    Note: This should NOT be used as an inverse of `to_token_sequence`. For
    that, use `from_token_sequence`.
    """
    # ys: head of every even-indexed token for every sequence in batch
    ys = toks[:, 0::2, :1]
    if not return_xs:
        return ys
    else:
        # xs: tail of every odd-indexed token for every sequence in batch
        xs = toks[:, 1::2, 1:]
        return ys, xs


def from_token_sequence(toks):
    """
    Inverse of `to_token_sequence`. Convert a sequence of vector tokens into
    a regression data set by decoding from a concatenating / joint encoding.
    
    Parameters:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded inputs and
        outputs.

    Returns:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.

    Note: This should NOT be used on the output of a transformer trained
    towards next-token prediction, since for such a transformer actually the
    ys are the result of predicting for the x tokens and vise versa. For that
    setting, use `from_predicted_token_sequence`.

    Example:
    
    ```
    > toks = [ [ [ 0, 1, 2, 3, ]       # 0, first x
               , [ 1, 0, 0, 0, ]       # first y, 0s
               , [ 0, 2, 3, 4, ]       # 0, second x
               , [ 2, 0, 0, 0, ]       # ...
               , [ 0, 6, 5, 4, ]
               , [ 6, 0, 0, 0, ]
               ] ]
    > xs, ys = from_token_sequence(toks)
    > xs
      [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    > ys
      [ [ [1],     [2],     [6],     ] ]
    """
    # xs: tail of every even-indexed token for every sequence in batch
    xs = toks[:, 0::2, 1:]
    # ys: head of every odd-indexed token for every sequence in batch
    ys = toks[:, 1::2, :1]
    return xs, ys

