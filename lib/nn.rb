require_relative "value.rb"

class Neuron

    def initialize(number_of_inputs)
        @initial_weights = Array.new(number_of_inputs) { rand(-1.0..1.0) }
        @initial_bias = rand(-1.0..1.0)

        @weights = @initial_weights.map { |w| Value.new(w) }
        @bias = Value.new(@initial_bias)
    end

    def reset_params
        @initial_weights.each_with_index do |w,i|
            @weights[i].value = w
        end
        @bias.value = @initial_bias
    end

    def set_params(params)
        n = 1 + @weights.size
        raise "Illegal number of parameters: #{params.size} expected #{n}" if n != params.size
        @bias.value = params[0]
        (1...params.size).each { |i| @weights[i - 1].value = params[i] }
    end

    attr_reader :weights, :bias

    def parameters
        self.weights + [self.bias]
    end
    
    def calc(inputs, activation)
        # xw + b
        n = self.weights.size
        raise "Wrong number of inputs! #{inputs.size} expected #{n}" unless n == inputs.size
        sum = self.bias
        n.times do |index|
            sum += self.weights[index] * inputs[index]
        end
        if activation == :tanh
            sum.tanh
        elsif activation == :relu
            sum.relu
        elsif activation == :sigmoid
            sum.sigmoid
        else
            raise "Unsupported activation function: #{activation}"
        end
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

    def reset_params
        self.neurons.each { |n| n.reset_params }
    end

    def calc(inputs, activation)
        outs = []
        self.neurons.each do |neuron|
            outs << neuron.calc(inputs, activation)
        end
        outs
    end
end

class MLP 
    
    def initialize(*layers_config)
        number_of_layers = layers_config.size
        @layers = Array.new(number_of_layers - 1) # input layer is not really a layer object
        (number_of_layers - 1).times do |i|
            @layers[i] = Layer.new(layers_config[i], layers_config[i + 1])
        end
        @layers_config = layers_config
    end

    attr_reader :layers

    def inspect
        "MLP(#{@layers_config.join(", ")})"
    end

    def parameters
        params = []
        self.layers.each { |layer| params += layer.parameters }
        params
    end

    def show_params(in_words = false)
        if in_words
            n = @layers_config[0]
            puts "Layer 0: (#{n} input#{n > 1 ? "s" : ""})"
            self.layers.each_with_index do |layer, i|
                n = layer.neurons.size
                puts "Layer #{i + 1}: (#{n} neuron#{n > 1 ? "s" : ""})"
                layer.neurons.each_with_index do |neuron, ii|
                    n = neuron.weights.size
                    puts "\tNeuron #{ii + 1}: (#{n} weight#{n > 1 ? "s" : ""})"
                    puts "\t\tBias: #{neuron.bias.value}"
                    w = neuron.weights.map { |v| v.value }.join(", ")
                    puts "\t\tWeights: #{w}"
                end
            end
        else
            n = @layers_config[0]
            self.layers.each_with_index do |layer, i|
                n = layer.neurons.size
                puts "["
                layer.neurons.each_with_index do |neuron, ii|
                    w = neuron.weights.map { |v| v.value }.join(", ")
                    puts "\t[ #{neuron.bias.value}, #{w} #{ii == layer.neurons.size - 1 ? ']' : '],'}"
                end
                puts i == self.layers.size - 1 ? "]" : "],"
            end
        end
        nil
    end

    def reset_params
        self.layers.each { |layer| layer.reset_params }
    end

    def set_params(params)
        params.each_with_index do |layer, li|
            layer.each_with_index do |neuron, ni|
                self.layers[li].neurons[ni].set_params(neuron)
            end
        end
    end

    def zero_grad
        self.parameters.each { |p| p.grad = 0.0 }
    end

    def calc(inputs, activation)
        out = inputs
        self.layers.each do |layer|
            out = layer.calc(out, activation) # chain the results forward, layer by layer
        end
        out.size == 1 ? out[0] : out # for convenience
    end
end
