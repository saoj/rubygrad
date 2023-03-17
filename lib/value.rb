
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
        other = Value.new(other) if !other.is_a?(Value)
        out = Value.new(self.value + other.value, '+', [self, other])

        backward = lambda do
            self.grad += out.grad
            other.grade += out.grad
        end

        return out
    end

end

a = Value.new(-4)
b = Value.new(2)
c = a + b
puts c.value
