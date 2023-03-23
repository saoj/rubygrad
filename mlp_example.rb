require 'rubygrad'
#require_relative 'lib/nn.rb'

# Build a Machine Learning Perceptron with 4 layers
# First  Layer (Layer 0) => Input Layer  => 3 Neurons => 3 Inputs
# Second Layer (Layer 1) => Hidden Layer => 4 Neurons
# Third  Layer (Layer 2) => Hidden Layer => 4 Neurons
# Fourth Layer (Layer 3) => Output Layer => 1 Neuron => 1 Output
nn = MLP.new(3, 4, 4, 1, :tanh)

nn.show_params
puts
nn.show_params(in_words = true)
puts

# 4 input samples
x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

# expected output for each of the 4 inputs above
y_expected = [1.0, -1.0, -1.0, 1.0]

passes = 2000
learning_rate = 0.2

(0...passes).each do |pass| 

    # forward pass (calculate output)
    y_calculated = x_inputs.map { |x| nn.calc(x) }

    # loss function (check how good the neural net is)
    loss = 0.0
    y_expected.each_index { |i| loss += (y_calculated[i] - y_expected[i]) ** 2 }

    # backward pass (calculate gradients)
    nn.zero_grad
    loss.backward

    # improve neural net (update weights and biases)
    nn.parameters.each { |p| p.value -= learning_rate * p.grad }

    nn.print_pass(learning_rate, loss, pass, passes) if (pass + 1) % 100 == 0 or pass == 0

    break if loss.value == 0 # just for fun and just in case
end

y_calculated = x_inputs.map { |x| nn.calc(x) }
puts
puts "Final NN results:"
y_calculated.each_with_index { |y_c, i| puts "Output: #{y_c} => Expected: #{y_expected[i]}" }
