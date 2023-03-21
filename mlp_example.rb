require_relative 'lib/nn.rb'

nn = MLP.new(3, 4, 4, 1)

nn.show_params(in_words = true)
puts

x_inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
y_expected = [1.0, -1.0, -1.0, 1.0] # desired

passes = 200
passes_format = "%#{passes.digits.length}d"
loss_precision = 10
loss_format = "%.#{loss_precision}f"
learning_rate = 0.2

(0...passes).each do |pass| 

    # forward pass
    y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }

    # loss function
    loss = 0.0
    y_expected.each_index { |i| loss += (y_calculated[i] - y_expected[i]) ** 2 }

    # backward pass
    nn.zero_grad
    loss.backward

    # improve
    nn.parameters.each { |p| p.value -= learning_rate * p.grad }

    puts "Pass #{passes_format % (pass + 1)} => Learning rate: #{"%.10f" % learning_rate} => Loss: #{loss_format % loss.value}" if (pass + 1) % 20 == 0 or pass == 0

    break if loss.value == 0 # just for fun just in case
end

y_calculated = x_inputs.map { |x| nn.calc(x, :tanh) }
puts
puts y_calculated
