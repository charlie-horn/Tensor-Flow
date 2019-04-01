class Operation():
    def __init__(self,input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass

class Placeholder():
    def __init__(self):
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)

class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)

class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default():
        global _default_graph
        _default_graph = self

