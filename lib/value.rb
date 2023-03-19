
class Value

    def initialize(value, op = '', prev = [])
        @value = value
        @op = op
        @grad = 0
        @prev = prev.uniq.freeze
        @calc_gradient = lambda { }
    end

    attr_reader :value, :grad, :op
    attr_writer :calc_gradient

    def +(other)
        other = to_v(other)
        out = Value.new(self.value + other.value, '+', [self, other])

        out.calc_gradient = lambda do
            self.grad += out.grad
            other.grade += out.grad
        end

        return out
    end

    def *(other)
        other = to_v(other)
        out = Value.new(self.value * other.value, '*', [self, other])

        out.calc_gradient = lambda do
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        end

        return out
    end

    def **(other)
        out = Value.new(self.value ** other, "**#{other}", [self])

        out.calc_gradient = lambda do
            self.grad += (other * self.value ** (other - 1)) * out.grad
        end

        return out
    end

    def tanh
        t = (Math.exp(2 * self.value) - 1) / (Math.exp(2 * self.value) + 1)
        out = Value.new(t, 'tanh', [self])

        out.calc_gradient = lambda do
            self.grad += (1 - t ** 2) * out.grad
        end

        return out
    end

    def -@
        self * -1
    end

    def -(other)
        self + (-other)
    end

    def /(other)
        self * (other ** -1)
    end

    def coerce(other)
        other = to_v(other)
        [other, self.value]
    end

    def to_s
        value.to_s
    end

    def inspect
        "Value(value=#{value}, grad=#{grad})"
    end

    private def to_v(other) = other.is_a?(Value) ? other : Value.new(other)

end

a = Value.new(1.0)
puts -a