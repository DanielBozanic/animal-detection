from functools import partial

class BackboneRouteExtractor():
    def __init__(self, backbone, indices=[3, 4]):
        self.__route_list = [0] * len(indices)
        self.__hooks = []
        self.__create_hooks(backbone, indices)

    def __create_hooks(self, backbone, indices):
        help_indices = list(range(len(indices)))
        for i, index in zip(indices, help_indices):
            function = partial(self.__hook_function, index=index)
            hook = backbone[i].register_forward_hook(function)
            self.__hooks.append(hook)

    def __hook_function(self, backbone, input, output, index):
        self.__route_list[index] = output

    @property
    def get_routes(self):
        return self.__route_list

    def remove_hooks(self):
        for x in self.__hooks:
            x.remove()
