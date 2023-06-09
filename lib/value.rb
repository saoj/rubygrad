require 'set'

class Value

    def initialize(value, prev = [])
        @value = value
        @grad = 0
        @prev = prev.uniq.freeze
        @calc_gradient = lambda { }
    end

    attr_reader :value, :grad, :prev, :calc_gradient
    attr_writer :calc_gradient, :grad, :value

    def +(other)
        other = to_v(other)
        out = Value.new(self.value + other.value, [self, other])

        out.calc_gradient = lambda do
            self.grad += out.grad
            other.grad += out.grad
        end

        return out
    end

    def *(other)
        other = to_v(other)
        out = Value.new(self.value * other.value, [self, other])

        out.calc_gradient = lambda do
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        end

        return out
    end

    def **(other)
        out = Value.new(self.value ** other, [self])

        out.calc_gradient = lambda do
            self.grad += (other * self.value ** (other - 1)) * out.grad
        end

        return out
    end

    def tanh
        t = (Math.exp(2.0 * self.value) - 1.0) / (Math.exp(2.0 * self.value) + 1.0)
        out = Value.new(t, [self])

        out.calc_gradient = lambda do
            self.grad += (1.0 - t ** 2.0) * out.grad
        end

        return out
    end

    def sigmoid
        e = Math.exp(-1.0 * self.value)
        t = 1.0 / (1.0 + e)
        out = Value.new(t, [self])
        
        out.calc_gradient = lambda do
            self.grad += t * (1.0 - t) * out.grad
        end

        return out
    end

    def relu
        n = self.value < 0 ? 0.0 : self.value
        out = Value.new(n, [self])

        out.calc_gradient = lambda do
            self.grad += (out.value > 0 ? 1.0 : 0.0) * out.grad
        end

        return out
    end

    def exp 
        out = Value.new(Math.exp(self.value), [self])

        out.calc_gradient = lambda do
            self.grad += out.value * out.grad
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

    def build_topo_graph(start)
        topo = []
        visited = Set.new
        build_topo = lambda do |v|
            if !visited.include?(v)
                visited.add(v)
                v.prev.each do |child|
                    build_topo.call(child)
                end
                topo.append(v)
            end
        end
        build_topo.call(start)
        return topo
    end

    def backward
        topo = build_topo_graph(self)
        self.grad = 1.0
        topo.reverse_each do |node|
            node.calc_gradient.call
        end
    end

    def to_s
        value.to_s
    end

    def inspect
        "Value(value=#{value}, grad=#{grad})"
    end

    private def to_v(other) = other.is_a?(Value) ? other : Value.new(other)

end


=begin
x1 = Value.new(2.0)
x2 = Value.new(0.0)
w1 = Value.new(-3.0)
w2 = Value.new(1.0)
b = Value.new(6.881373587)
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh
o.backward

puts x1.inspect,x2.inspect,w1.inspect,w2.inspect

=end


