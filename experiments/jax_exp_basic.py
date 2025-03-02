from model import JAX_SimpleBaseline


class JAX_Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'JAX_SimpleBaseline': JAX_SimpleBaseline
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass