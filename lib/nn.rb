require_relative "value.rb"

class Neuron

    def initialize(number_of_inputs)
        @weights = Array.new(number_of_inputs) { Value.new(rand(-1.0..1.0)) }
        @bias = Value.new(rand(-1.0..1.0)) # bias is only one per neuron
    end

    attr_reader :weights, :bias
    
    def calc(inputs)
        # xw + b
        n = self.weights.size
        raise "Wrong number of inputs! #{inputs.size} expected #{n}" unless n == inputs.size
        sum = self.bias
        n.times do |index|
            sum += self.weights[index] * inputs[index]
        end
        sum
    end
end

class Layer

    def initialize(number_of_inputs, number_of_outputs)
        @neurons = Array.new(number_of_outputs) { Neuron.new(number_of_inputs) }
    end

    attr_reader :neurons

    def calc(inputs)
        outs = []
        self.neurons.each do |neuron|
            outs << neuron.calc(inputs)
        end
        outs
    end

end

class MLP 
    
    def initialize(layer_config)
        number_of_layers = layer_config.size
        @layers = Array.new(number_of_layers - 1) # input layer is not really a layer object
        (number_of_layers - 1).times do |i|
            @layers[i] = Layer.new(layer_config[i], layer_config[i + 1])
        end
    end

    attr_reader :number_of_layers, :layers

    def calc(inputs)
        out = inputs
        self.layers.each do |layer|
            out = layer.calc(out) # chain the results forward, layer by layer
        end
        out
    end
    
end

x = [2.0, 3.0, -1.0]
nn = MLP.new([3, 4, 4, 1])
puts nn.calc(x).inspect
