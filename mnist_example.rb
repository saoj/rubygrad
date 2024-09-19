###
# ATTENTION: This is very slow. Not using parallel processing. Not using GPU. Not using batch processing.
# Needs to find a way to improve runtime.
### 

require_relative 'lib/nn.rb'
require 'mnist-ready'

# Build a Neural Network for MNIST digit classification
# Input Layer  => 784 Neurons (28x28 pixel images)
# Hidden Layer => 128 Neurons
# Output Layer => 10 Neurons (one for each digit 0-9)
nn = MLP.new(784, 128, 10, :relu)
puts
puts "#{nn} => #{nn.param_count} parameters"
puts

# Assuming we have these datasets
MNIST = MnistDataset.instance(show_progress = true)

train_digits = MNIST.train_set
test_digits = MNIST.test_set

# Training parameters
passes = 10000
learning_rate = 0.01

(0...passes).each do |pass|

  total_loss = Value.new(0.0)

  train_digits.each do |digit|
    # Forward pass
    output = nn.calc(digit.pixels)

    # Calculate loss (cross-entropy loss)
    expected = Array.new(10, 0.0)
    expected[digit.label] = 1.0
    loss = Value.new(0.0)
    epsilon = 1e-15
    output.each_with_index do |o, i|
      loss -= expected[i] * Math.log([o.value, epsilon].max) + (1 - expected[i]) * Math.log([1 - o.value, epsilon].max)
    end
    total_loss += loss

    # Backward pass
    nn.zero_grad
    loss.backward

    # Update weights and biases
    nn.parameters.each { |p| p.value -= learning_rate * p.grad }

    # Print progress every 20 digits
    if (train_digits.index(digit) + 1) % 20 == 0
      puts "Processed #{train_digits.index(digit) + 1} digits in pass #{pass + 1}"
    end
  end

  # Print progress
  if (pass + 1) % (passes / 20) == 0 || pass == 0
    avg_loss = total_loss.value / train_digits.size
    puts "Pass #{pass + 1}/#{passes}, Avg Loss: #{avg_loss}"
  end
end

# Evaluate on test set
correct = 0
test_digits.each do |digit|
  output = nn.calc(digit.pixels)
  predicted = output.index(output.max)
  correct += 1 if predicted == digit.label
end

accuracy = correct.to_f / test_digits.size
puts "Correct guesses: #{correct}"
puts "Total guesses: #{test_digits.size}"
puts "Test Accuracy: #{accuracy * 100}%"
