defmodule GradientDescentTest do
  use ExUnit.Case
  import GradientDescent

  @delta 1.0e-2

  defmacrop assert_delta(lhs, rhs) do
    quote do
      assert_in_delta unquote(lhs), unquote(rhs), unquote(@delta)
    end
  end

  test "cost" do
    assert_delta(cost([0], [0], 0, 0), 0)
    assert_delta(cost([0], [0], 0, 0), 0)
    assert_delta(cost([1], [0], 0, 0), 0)
    assert_delta(cost([1], [1], 0, 0), 0.5)
    assert_delta(cost([1], [1], 1, 0), 0)
    assert_delta(cost([1], [1], 1, 1), 0.5)
    assert_delta(cost([1, 2, 3], [1, 2, 3], 1, 1), 0.5)
    assert_delta(cost([1, 2, 3], [1, 2, 3], 3, 4), 33.3333333)
    assert_delta(cost([1, 2, 3], [3, 2, 1], 3, 4), 37.3333333)
  end

  defmacrop assert_tuple_deltas(lhs, rhs) do
    quote do
      unquote(lhs)
      |> Tuple.to_list()
      |> Enum.zip(Tuple.to_list(unquote(rhs)))
      |> Enum.each(fn {l, r} ->
        assert_delta(l, r)
      end)
    end
  end

  test "gradient" do
    assert_tuple_deltas(gradient([0], [0], 0, 0), {0, 0})
    assert_tuple_deltas(gradient([0, 0, 0], [0, 0, 0], 0, 0), {0, 0})
    assert_tuple_deltas(gradient([0, 1, 2], [0, 1, 2], 0, 0), {-1.66666666, -1.0})
    assert_tuple_deltas(gradient([0, 1, 2], [0, 1, 2], 2, 1), {2.66666666, 2.0})
    assert_tuple_deltas(gradient([0, 1, 2], [0, 1, 2], 1, 2), {2.0, 2.0})
    assert_tuple_deltas(gradient([0, 1, 2], [2, 1, 0], 1, 2), {3.33333333, 2.0})
    assert_tuple_deltas(gradient([0, 1, 2], [2, 1, 0], 3, 4), {8.66666666, 6.0})
  end

  test "gradient iteration" do
    assert_tuple_deltas(gradient_iteration({0, 0}, [0], [0], 1), {0, 0})
    assert_tuple_deltas(gradient_iteration({1, 2}, [0, 1, 2], [0, 1, 2], 1), {-1.0, 0.0})
    assert_tuple_deltas(gradient_iteration({1, 2}, [0, 1, 2], [2, 1, 0], 1), {-2.333333, 0.0})
    assert_tuple_deltas(gradient_iteration({1, 2}, [0, 1, 2], [0, 1, 2], 0.01), {0.98, 1.98})
  end

  test "gradient descent" do
    assert {wb, _} = gradient_descent({0, 0}, [1.0, 2.0], [300.0, 500.0], 1.0e-2, 10_000)

    assert_tuple_deltas(wb, {199.9929, 100.0016})
  end
end
