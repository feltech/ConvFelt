Feature: Executing a single backprop
  As a middleware developer
  I want to be able to execute a single training iteration on a single image
  Because then I can define workflows that build on it

  Background:
    Given a model defined by the following YAML
    """
    - type: input
      size:
        width: 128
        height: 128
        channels: 4
    - type: conv
      filter:
        input:
          width: 3
          height: 3
        output:
          width: 1
          height: 1
          channels: 6
    - type: relu
    - type: fully_connected
      channels: 1024
    - type: relu
    - type: fully_connected
      channels: 2
    - type: softmax
    """
    And model is initialised with random weights

  Scenario: Exported structure matches imported structure
    When YAML is generated from the network structure
    Then input YAML matches output YAML

  Scenario: A single inference
    When inference is performed on the image
    Then the result is in [0, 1] for each class
    And the sum across classes is 1

  Scenario: Repeated inferences give the same result
    When inference is performed on the image 2 times
    Then the inferred values haven't changed

  Scenario: A single backprop
    Given the correct class for the image is 1
    And inference is performed on the image
    When backprop is used to update the weights
    And inference is performed on the image
    Then the value for class 1 has increased
