Gem::Specification.new do |s|
    s.name        = 'rubygrad'
    s.version     = '1.2.4'
    s.licenses    = ['MIT']
    s.summary     = "A port of Andrej Karpathy's micrograd to Ruby."
    s.authors     = ["Sergio Oliveira Jr"]
    s.email       = 'sergio.oliveira.jr@gmail.com'
    s.files       = Dir['lib/**/*.rb'] + Dir['*.rb']
    s.homepage    = 'https://github.com/saoj/rubygrad'
    s.required_ruby_version = '>= 3.0.0'
  end
