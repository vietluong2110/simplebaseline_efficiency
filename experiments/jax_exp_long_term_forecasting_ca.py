
from data_provider.data_factory import data_provider
from experiments.jax_exp_basic import JAX_Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import os
import time
import warnings
import numpy as np


from flax import nnx 
import optax
import jax.numpy as jnp 
import jax
import orbax.checkpoint as ocp

warnings.filterwarnings('ignore')

class JAX_Exp_Long_Term_Forecast(JAX_Exp_Basic):
    def __init__(self, args):
        super(JAX_Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x
            batch_y = batch_y

            if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'glad' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark
                batch_y_mark = batch_y_mark

            # decoder input
            dec_inp = jnp.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = jnp.concatenate((batch_y[:, :self.args.label_len, :], dec_inp), axis=1)
            
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if self.args.features == 'MS' else 0

            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs
            true = batch_y

            if self.args.data == 'PEMS':
                B, T, C = pred.shape
                pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                mae, mse, rmse, mape, mspe = metric(pred, true)
                total_loss.append(mae)
            else:
                loss = jnp.mean(optax.losses.squared_error(outputs, batch_y))
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    

    def train(self, setting):

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        # if not os.path.exists(path):
        #     os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        total_steps = train_steps * self.args.train_epochs  # total training steps
        lr_scheduler = optax.schedules.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=self.args.learning_rate,
            pct_start=self.args.pct_start
        )
                
        optimizer = nnx.Optimizer(self.model, optax.adamw(learning_rate=lr_scheduler))

        train_step_jit = jax.jit(train_step, static_argnames=('pred_len', 'l1_weight', 'output_attention', 'data', 'features'))
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                batch_x = batch_x
                batch_y = batch_y
                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'glad' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark
                    batch_y_mark = batch_y_mark

                # decoder input
                dec_inp = jnp.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = jnp.concatenate((batch_y[:, :self.args.label_len, :], dec_inp), axis=1)

                graphdef, state = nnx.split((self.model, optimizer))

                batch_x = batch_x.astype(np.float32)

                state, loss = train_step_jit(graphdef, state, self.args.pred_len, self.args.l1_weight, self.args.output_attention, self.args.data, self.args.features
                                  , batch_x, batch_x_mark,dec_inp, batch_y_mark, batch_y)

                nnx.update((self.model, optimizer), state)
                train_loss.append(loss)
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = jnp.average(jnp.array(train_loss))
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # _, self.state = nnx.split(self.model)
        # self.checkpoint_dir = '/workspace/save_1'
        # checkpoint_manager = ocp.CheckpointManager(
        # ocp.test_utils.erase_and_create_empty(self.checkpoint_dir),
        #     options=ocp.CheckpointManagerOptions(
        #         max_to_keep=2,
        #         keep_checkpoints_without_metrics=False,
        #         enable_async_checkpointing=False,
        #         create=True,
        #     ),
        # )

        # checkpoint_manager.save(
        #     1, args=ocp.args.Composite(state=ocp.args.PyTreeSave(self.state))
        # )
        # checkpoint_manager.wait_until_finished()
        # checkpoint_manager.close()
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     # Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
        #     abstract_model = nnx.eval_shape(lambda: self._build_model())
        #     graphdef, abstract_state = nnx.split(abstract_model)

        #     # state_restored = checkpointer.restore(self.ckpt_dir /'state', abstract_state)
        #     with ocp.CheckpointManager(
        #         self.checkpoint_dir, options=ocp.CheckpointManagerOptions(read_only=True)
        #     ) as read_mgr:
        #         restored = read_mgr.restore(
        #             1,
        #             # pass in the model_state to restore the exact same State type
        #             args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=abstract_state))
        #         )
        #         read_mgr.wait_until_finished()

        #     # The model is now good to use!
        #     model = nnx.merge(graphdef, restored['state'])

        preds = []
        trues = []
        folder_path = './checkpoints/' + setting + '/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

            batch_x = batch_x
            batch_y = batch_y

            if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'glad' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark
                batch_y_mark = batch_y_mark

            # decoder input
            dec_inp = jnp.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = jnp.concatenate((batch_y[:, :self.args.label_len, :], dec_inp), axis=1)
            # encoder - decoder
            
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                
                outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
            outputs = outputs
            batch_y = batch_y       

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)
            if i % 20 == 0:
                input = batch_x
                if test_data.scale and self.args.inverse:
                    shape = input.shape
                    input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)


        # result save
        folder_path = './checkpoints/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        # if load:
        #     path = os.path.join(self.args.checkpoints, setting)
        #     best_model_path = path + '/' + 'checkpoint.pth'
        #     self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x
            batch_y = batch_y
            batch_x_mark = batch_x_mark
            batch_y_mark = batch_y_mark

            # decoder input
            dec_inp = jnp.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = jnp.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)
            # encoder - decoder
            if self.args.use_amp:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs
            if pred_data.scale and self.args.inverse:
                shape = outputs.shape
                outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
    def benchmark(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        folder_path = './checkpoints/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()

        flax_apply_jitted = jax.jit(lambda batch_x, batch_x_mark, dec_inp, batch_y_mark: self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))

        sum_time = 0; 
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

            if i == 0: 
                print("Warm up")
            t1 = time.time()


            batch_x = batch_x
            batch_y = batch_y
            
            if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'glad' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark
                batch_y_mark = batch_y_mark

            # decoder input
            dec_inp = jnp.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = jnp.concatenate((batch_y[:, :self.args.label_len, :], dec_inp), axis=1)
            # encoder - decoder

            flax_apply_jitted(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            
            if i != 0:
                sum_time += time.time() - t1
            print(f"eager eval time {i}: {time.time() - t1}")
            print("~" * 10)
        print(f"average eager eval time {i}: {sum_time / (len(test_loader) - 1)}")
        return

    # Returns the result of running `fn()` and the time it took for `fn()` to run,
    # in seconds. We use CUDA events and synchronization for the most accurate
    # measurements.
    def timed(self, fn):
        start = time.time()
        result = fn()
        end = time.time()
        return result, end - start

def loss_fn(model_, batch_y_, pred_len, l1_weight, output_attention, data, features, batch_x, batch_x_mark, dec_inp, batch_y_mark):

    outputs, attn  = model_(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    f_dim = -1 if features == 'MS' else 0                        
    outputs = outputs[:, -pred_len:, f_dim:]
    batch_y_ = batch_y_[:, -pred_len:, f_dim:]
    loss = jax.lax.cond(
        data == "PEMS",
        lambda _: jnp.mean(jnp.abs(outputs - batch_y_)) + l1_weight * jnp.mean(jnp.abs(jnp.stack(attn))),
        lambda _: jnp.mean(optax.losses.squared_error(outputs, batch_y_)) + l1_weight * jnp.mean(jnp.abs(jnp.stack(attn))),
        operand=None
    )
    
    return loss
    
def train_step(graphdef, state, pred_len, l1_weight, output_attention, data, features, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y):
        model, optimizer = nnx.merge(graphdef, state)
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch_y, pred_len, l1_weight, output_attention, data, features, batch_x, batch_x_mark, dec_inp, batch_y_mark)
        optimizer.update(grads)
        _, state = nnx.split((model, optimizer))
        return state, loss 