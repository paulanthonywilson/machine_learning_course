defmodule GradientDescent do
  @moduledoc """
  Keep this naively close to the source python and calculate in Elixir.

  If appropriate I'll take the opportunity to explore Nx.
  """

  @spec cost(
          features :: list(float()),
          targets :: list(float()),
          slope :: float(),
          bias :: float()
        ) :: float()
  def cost(features, targets, w, b) do
    m = length(features)

    features
    |> Enum.zip(targets)
    |> Enum.reduce(0, fn {x, y}, acc ->
      y_hat = w * x + b
      diff_squared = :math.pow(y_hat - y, 2)
      acc + diff_squared / (2 * m)
    end)
  end

  @spec gradient(
          features :: list(float()),
          targets :: list(float()),
          slope :: float(),
          bias :: float()
        ) :: {d_slope :: float(), d_bias :: float()}
  def gradient(features, targets, w, b) do
    m = length(features)

    {dw, db} =
      features
      |> Enum.zip(targets)
      |> Enum.map(fn {x, y} ->
        y_dist = x * w + b - y
        {y_dist * x, y_dist}
      end)
      |> Enum.reduce(fn {dw_i, db_i}, {dw, db} ->
        {dw + dw_i, db + db_i}
      end)

    {dw / m, db / m}
  end

  @spec gradient_iteration(
          {slope :: float(), bias :: float()},
          features :: list(float()),
          targets :: list(float()),
          alpha :: float()
        ) :: {slope :: float(), bias :: float()}
  def gradient_iteration({w, b}, features, target, alpha) do
    {dw, db} = gradient(features, target, w, b)
    {w - dw * alpha, b - db * alpha}
  end

  def gradient_descent(wb, features, targets, alpha, iterations, history \\ [])

  def gradient_descent({_w, _b} = wb, _features, _targets, _alpha, 0, history) do
    {wb, history}
  end

  def gradient_descent({_w, _b} = wb, features, targets, alpha, iteration, history) do
    wb
    |> gradient_iteration(features, targets, alpha)
    |> gradient_descent(features, targets, alpha, iteration - 1, [wb | history])
  end
end
