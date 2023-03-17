
class Value

    def initialize(value, op = '', prev = [])
        @value = value
        @op = op
        @grad = 0
        @prev = prev.uniq.freeze
        @backward = nil
    end

    attr_reader :value, :grad, :backward, :op

    def +(other)
        other = to_v(other)
        out = Value.new(self.value + other.value, '+', [self, other])

        backward = lambda do
            self.grad += out.grad
            other.grade += out.grad
        end

        return out
    end

    def *(other)
        other = to_v(other)
        out = Value.new(self.value * other.value, '*', [self, other])

        backward = lambda do
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        end

        return out
    end

    def to_s
        value.to_s
    end

    def inspect
        "Value(value=#{value}, grad=#{grad})"
    end

    private def to_v(other) = other.is_a?(Value) ? other : Value.new(other)

end

a = Value.new(-4)
b = Value.new(2)
c = a + b
d = a * b
puts c.value
puts d.value

puts c
puts c.inspect
c.blah