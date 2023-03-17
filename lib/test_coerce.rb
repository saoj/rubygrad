
class MyValue

    def initialize(value)
        @value = value
    end

    attr_reader :value

    private def to_v(other) = other.is_a?(MyValue) ? other : MyValue.new(other)

    def +(other)
        other = to_v(other)
        out = MyValue.new(self.value + other.value)
    end

    def *(other)
        other = to_v(other)
        out = MyValue.new(self.value * other.value)
    end

    def **(other)
        out = MyValue.new(self.value ** other)
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

end

a = MyValue.new(4.0) # a MyValue
b = MyValue.new(2.0) # a MyValue
c = 2.0 # an Integer 

res = a - b
puts "#{res} is #{res.class}" # => 2.0 is MyValue

res = b - a
puts "#{res} is #{res.class}" # => -2.0 is MyValue

res = a - c
puts "#{res} is #{res.class}" # => 2.0 is MyValue

res = c - a
puts "#{res} is #{res.class}" # => -2.0 is MyValue

puts

res = a / b
puts "#{res} is #{res.class}" # => 2.0 is MyValue

res = b / a
puts "#{res} is #{res.class}" # => 0.5 is MyValue

res = a / c
puts "#{res} is #{res.class}" # => 2.0 is MyValue

res = c / a
puts "#{res} is #{res.class}" # => 0.5 is MyValue


