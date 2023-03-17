
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
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        end

        return out
    end

    private

    def to_v(other)
        other.is_a?(Value) ? other : Value.new(other)
    end

end

a = Value.new(-4)
b = Value.new(2)
c = a + b
d = a * b
puts c.value
puts d.value
