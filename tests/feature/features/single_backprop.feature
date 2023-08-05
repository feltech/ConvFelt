Feature: Executing a single backprop
  As a middleware developer
  I want to be able to execute a single training iteration on a single image
  Because then I can define workflows that build on it

  Background:
    Given image 'resources/plus.png'
    And a 'INPUT - CONV-42-3x3x1-1 - RELU - FC-1024 - RELU - FC-2' model
    And model is initialised with random weights

  Scenario: A single inference
    When inference is performed on the image
    Then the result is in ['0', '1'] for each class
    And the sum across classes is 1
    And repeated inferences give the same result

  Scenario: A single backprop
    Given the correct class for the image is 1
    When backprop is used to update the weights
    Then the value for class 1 is greater after backprop than before
