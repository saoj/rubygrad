require_relative "value.rb"

class Neuron

    def initialize(number_of_inputs)
        @weights = Array.new(number_of_inputs) { Value.new(rand(-1.0..1.0)) }
        @bias = Value.new(rand(-1.0..1.0)) # bias is only one per neuron
    end

    attr_reader :weights, :bias

    def parameters
        self.weights + [self.bias]
    end
    
    def calc(inputs)
        # xw + b
        n = self.weights.size
        raise "Wrong number of inputs! #{inputs.size} expected #{n}" unless n == inputs.size
        sum = self.bias
        n.times do |index|
            sum += self.weights[index] * inputs[index]
        end
        sum.tanh
    end
end

class Layer

    def initialize(number_of_inputs, number_of_outputs)
        @neurons = Array.new(number_of_outputs) { Neuron.new(number_of_inputs) }
    end

    attr_reader :neurons

    def parameters
        params = []
        self.neurons.each { |n| params += n.parameters }
        params
    end

    def calc(inputs)
        outs = []
        self.neurons.each do |neuron|
            outs << neuron.calc(inputs)
        end
        outs
    end

end

class MLP 
    
    def initialize(*layer_config)
        number_of_layers = layer_config.size
        @layers = Array.new(number_of_layers - 1) # input layer is not really a layer object
        (number_of_layers - 1).times do |i|
            @layers[i] = Layer.new(layer_config[i], layer_config[i + 1])
        end
    end

    attr_reader :number_of_layers, :layers

    def parameters
        params = []
        self.layers.each { |layer| params += layer.parameters }
        params
    end

    def calc(inputs)
        out = inputs
        self.layers.each do |layer|
            out = layer.calc(out) # chain the results forward, layer by layer
        end
        out.size == 1 ? out[0] : out # for convenience
    end
    
end

nn = MLP.new(3, 4, 4, 1)

x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y_expected = [1.0, -1.0, -1.0, 1.0] # desired

(1..20).each do |pass| 

    # forward pass
    y_calculated = x_inputs.map { |x| nn.calc(x) }

    # loss function
    loss = 0.0
    y_expected.each_index { |i| loss += (y_calculated[i] - y_expected[i]) ** 2 }

    # backward pass
    loss.backward

    # improve
    nn.parameters.each { |p| p.value += -0.01 * p.grad }

    puts "Pass #{pass} => Loss: #{loss.value}"
end
