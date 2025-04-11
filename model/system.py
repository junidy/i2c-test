from argparse import ArgumentParser
import torch
import torchaudio
import auraloss
from dasp_pytorch import ParametricEQ, Compressor
# from automix.utils import restore_from_0to1
import pytorch_lightning as pl

def restore_from_0to1(x: torch.Tensor, min_val: float, max_val: float):
    """Restore tensor back to the original range assuming they have been normalized on (0,1)

    Args:
        x (torch.Tensor): Tensor with normalized values on (0,1).
        min_val (float): Minimum value in the original range.
        max_val (float): Maximum value in the original range.

    Returns:
        y (torch.Tensor): Tensor with denormalized values on (min_val, max_val).
    """
    return (x * (max_val - min_val)) + min_val

class VGGishEncoder(torch.nn.Module):
    def __init__(self, sample_rate: float) -> None:
        super().__init__()
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        model.eval()
        self.sample_rate = sample_rate
        self.model = model
        self.d_embed = 128
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        bs, seq_len = x.size()
        with torch.no_grad():
            if self.sample_rate != 16000:
                x = self.resample(x)
            z = []
            for bidx in range(bs):
                x_item = x[bidx : bidx + 1, :]
                x_item = x_item.permute(1, 0)
                x_item = x_item.cpu().view(-1).numpy()
                z_item = self.model(x_item, fs=16000)
                z_item = z_item.mean(dim=0)  # mean across time frames
                z.append(z_item)
            z = torch.cat(z, dim=0)
        return z


class Res_2d(torch.nn.Module):
    """Residual 2D Convolutional layer.

    Args:
        input_channels (int):

    Adapted from https://github.com/minzwon/sota-music-tagging-models. Licensed under MIT by Minz Won.
    """

    def __init__(self, input_channels: int, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = torch.nn.Conv2d(
            input_channels,
            output_channels,
            shape,
            stride=stride,
            padding=shape // 2,
        )
        self.bn_1 = torch.nn.BatchNorm2d(output_channels)
        self.conv_2 = torch.nn.Conv2d(
            output_channels,
            output_channels,
            shape,
            padding=shape // 2,
        )
        self.bn_2 = torch.nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = torch.nn.Conv2d(
                input_channels,
                output_channels,
                shape,
                stride=stride,
                padding=shape // 2,
            )
            self.bn_3 = torch.nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class ShortChunkCNN_Res(torch.nn.Module):
    """Short-chunk CNN architecture with residual connections.

    Args:
        sample_rate (float): Audio input sampling rate.
        n_channels (int): Number of convolutional channels. Default: 128
        n_fft (int): FFT size for computing melspectrogram. Default: 1024
        n_mels (int): Number of mel frequency bins: Default 128

    Adapted from https://github.com/minzwon/sota-music-tagging-models. Licensed under MIT by Minz Won.
    """

    def __init__(
        self,
        sample_rate,
        n_channels=128,
        n_fft=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=50,
        ckpt_path: str = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = torch.nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        # Dense
        self.dense1 = torch.nn.Linear(n_channels * 4, n_channels * 4)
        self.bn = torch.nn.BatchNorm1d(n_channels * 4)
        self.dense2 = torch.nn.Linear(n_channels * 4, n_class)
        self.dropout = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()

        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(checkpoint)
            print(f"Loaded weights from {ckpt_path}")

        self.d_embed = n_channels * 4
        self.resample = torchaudio.transforms.Resample(sample_rate, 16000)

    def forward(self, x):

        # resampling
        if self.sample_rate != 16000:
            x = self.resample(x)

        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = torch.nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        # x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.dense2(x)
        # x = nn.Sigmoid()(x)

        return x


class PostProcessor(torch.nn.Module):
    def __init__(self, num_params: int, d_embed: int) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_embed, 256),
            torch.nn.Dropout(0.2),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(0.2),
            torch.nn.PReLU(),
            torch.nn.Linear(256, num_params),
            torch.nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        return self.mlp(z)


class Mixer(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_dB: int = -48.0,
        max_gain_dB: int = 24.0,
    ) -> None:
        super().__init__()
        self.eq = ParametricEQ(sample_rate)
        self.comp = Compressor(sample_rate)

        self.num_params = 2 + self.eq.num_params + self.comp.num_params
        self.param_names = ["Gain dB"] + list(self.eq.param_ranges.keys()) + list(self.comp.param_ranges.keys()) +  ["Pan"]
        self.sample_rate = sample_rate
        self.min_gain_dB = min_gain_dB
        self.max_gain_dB = max_gain_dB

    # ---------------------EDIT THIS--------------------------------------------
    def forward(self, x: torch.Tensor, p: torch.Tensor):
        """Generate a mix of stems given mixing parameters normalized to (0,1).

        Args:
            x (torch.Tensor): Batch of waveform stem tensors with shape (bs, num_tracks, seq_len).
            p (torch.Tensor): Batch of normalized mixing parameters (0,1) for each stem with shape (bs, num_tracks, num_params)

        Returns:
            y (torch.Tensor): Batch of stereo waveform mixes with shape (bs, 2, seq_len)
        """
        bs, num_tracks, seq_len = x.size()

        gain_param = p[..., 0]
        eq_params = p[..., 1:1+self.eq.num_params]
        comp_params = p[..., 1+self.eq.num_params:-1]
        pan_param = p[..., -1]


        # ------------- apply gain -------------
        gain_dB = p[..., 0]  # get gain parameter
        gain_dB = restore_from_0to1(gain_dB, self.min_gain_dB, self.max_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)  # convert gain from dB scale to linear
        gain_lin = gain_lin.view(bs, num_tracks, 1)  # reshape for multiplication
        x = x * gain_lin  # apply gain (bs, num_tracks, seq_len)

        # ------------- apply EQ -------------
        eq_param_dict = self.eq.denormalize_param_dict(
            self.eq.extract_param_dict(eq_params.view(-1, self.eq.num_params))
        )
        x = x.view(-1, 1, seq_len)  # Reshape for EQ processing
        x = self.eq.process_fn(x, self.sample_rate, **eq_param_dict)
        x = x.view(bs, num_tracks, seq_len)

        # ------------- apply Compression -------------
        comp_param_dict = self.comp.denormalize_param_dict(
            self.comp.extract_param_dict(comp_params.view(-1, self.comp.num_params))
        )
        x = x.view(-1, 1, seq_len)  # Reshape for compression processing
        x = self.comp.process_fn(x, self.sample_rate, **comp_param_dict)
        x = x.view(bs, num_tracks, seq_len)


        # ------------- apply panning -------------
        # expand mono stems to stereo, then apply panning
        x = x.view(bs, num_tracks, 1, -1)  # (bs, num_tracks, 1, seq_len)
        x = x.repeat(1, 1, 2, 1)  # (bs, num_tracks, 2, seq_len)

        pan = p[..., 1]  # get pan parameter
        pan_theta = pan * torch.pi / 2
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=-1)
        pan_gains_lin = pan_gains_lin.view(bs, num_tracks, 2, 1)  # reshape for multiply
        x = x * pan_gains_lin  # (bs, num_tracks, 2, seq_len)

        # ----------------- apply mix -------------
        # generate a mix for each batch item by summing stereo tracks
        y = torch.sum(x, dim=1)  # (bs, 2, seq_len)


         # Stack processed parameters
        processed_params = [
            gain_dB.view(bs, num_tracks, 1),
        ]

        for param_name in self.eq.param_ranges.keys():
            processed_params.append(
                eq_param_dict[param_name].view(bs, num_tracks, 1)
            )

        for param_name in self.comp.param_ranges.keys():
            processed_params.append(
                comp_param_dict[param_name].view(bs, num_tracks, 1)
            )

        processed_params.append(pan_param.view(bs, num_tracks, 1))

        # --- Add Debug Print HERE ---
        print(f"DEBUG: Mixer.forward returning types: type(y)={type(y)}, type(p)={type(p)}")
        print(f"DEBUG: Mixer.forward returning y.shape={y.shape if isinstance(y, torch.Tensor) else 'N/A'}, p.shape={p.shape if isinstance(p, torch.Tensor) else 'N/A'}")
        # --- End Debug Print ---

        return y, p




class DifferentiableMixingConsole(torch.nn.Module):
    """Differentiable mixing console.

    Notes:
        We do not use neural audio effect proxies as in the original publication.
        Instead we use a set of explicitly differentiable audio effects.

    Steinmetz et al. (2021). Automatic multitrack mixing with a differentiable mixing console of neural audio effects. ICASSP.
    """

    def __init__(
        self,
        sample_rate: int,
        encoder_arch: str = "short_res",
        load_weights: bool = False,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder_arch = encoder_arch

        # Creates a mix given tracks and parameters (also called the "Transformation Network")
        self.mixer = Mixer(sample_rate)

        # Simple 2D CNN on spectrograms
        if encoder_arch == "vggish":
            self.encoder = VGGishEncoder(sample_rate)
        elif encoder_arch == "short_res":
            self.encoder = ShortChunkCNN_Res(
                sample_rate,
                ckpt_path="./models/encoder.ckpt" if load_weights else None,
            )
        else:
            raise ValueError(f"Invalid encoder_arch: {encoder_arch}")

        # MLP projects embedding + context to parameter space
        self.post_processor = PostProcessor(
            self.mixer.num_params,
            self.encoder.d_embed * 2,
        )

    def block_based_forward(
        self,
        x: torch.Tensor,
        block_size: int,
        hop_length: int,
    ):
        bs, num_tracks, seq_len = x.size()

        x = torch.nn.functional.pad(
            x,
            (block_size // 2, block_size // 2),
            mode="reflect",
        )

        unfold_fn = torch.nn.Unfold(
            (1, block_size),
            stride=(1, hop_length),
        )
        fold_fn = torch.nn.Fold(
            (1, x.shape[-1]),
            (1, block_size),
            stride=(1, hop_length),
        )
        window = torch.hann_window(block_size)
        window = window.view(1, 1, -1)
        window = window.type_as(x)

        x_track_blocks = []
        for track_idx in range(num_tracks):
            x_track_blocks.append(unfold_fn(x[:, track_idx, :].view(bs, 1, 1, -1)))

        x_blocks = torch.stack(x_track_blocks, dim=1)
        num_blocks = x_blocks.shape[-1]

        block_outputs = []
        for block_idx in range(num_blocks):
            x_block = x_blocks[..., block_idx]
            x_result, _ = self.forward(x_block)
            x_result = x_result.view(bs, 1, 2, -1)
            block_outputs.append(x_result)

        block_outputs = torch.cat(block_outputs, dim=1)
        block_outputs = block_outputs * window  # apply overlap-add window
        y_left = fold_fn(block_outputs[:, :, 0, :].permute(0, 2, 1))
        y_right = fold_fn(block_outputs[:, :, 1, :].permute(0, 2, 1))
        y = torch.cat((y_left, y_right), dim=1)

        # crop the padded areas
        y = y[..., block_size // 2 : -(block_size // 2)]

        return y

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None):
        """Given a set of tracks, analyze them with a shared encoder, predict a set of mixing parameters,
        and use these parameters to generate a stereo mixture of the inputs.

        Args:
            x (torch.Tensor): Input tracks with shape (bs, num_tracks, channels, seq_len) # Corrected shape comment
            track_mask (torch.Tensor, optional): Mask specifying inactivate tracks with shape (bs, num_tracks)

        Returns:
            y (torch.Tensor): Final stereo mixture with shape (bs, 2, seq_len)
            p (torch.Tensor): Estimated NORMALIZED mixing parameters with shape (bs, num_tracks, num_params) # Corrected return description
        """
        # --- CORRECTED UNPACKING ---
        # Unpack all 4 dimensions
        bs, num_tracks, num_channels, seq_len = x.size()
        print(f"DEBUG: DMC received input shape: {(bs, num_tracks, num_channels, seq_len)}") # Log received shape
        # --- END CORRECTION ---


        # if no track_mask supplied assume all tracks active
        if track_mask is None:
            # Ensure mask has correct dimensions based on input
            track_mask = torch.zeros(bs, num_tracks).type_as(x).bool() # Mask is (bs, num_tracks)

        # move tracks AND channels to the batch dimension for shared encoder
        # Original code assumed 3D input, adjust for 4D
        # x = x.view(bs * num_tracks, -1) # Original, incorrect for 4D
        x = x.reshape(bs * num_tracks, num_channels, seq_len) # Reshape to (bs*num_tracks, channels, seq_len)
        # IMPORTANT: Ensure your encoder can handle this shape!

        # --- Prepare input for MONO Encoder ---
        # IMPORTANT: Assuming self.encoder (ShortChunkCNN_Res) expects MONO input [batch, seq_len]
        if num_channels == 1:
            print(f"DEBUG: Encoder expects MONO. Squeezing channel dim {num_channels} for encoder input.")
            # Reshape from (bs*num_tracks, 1, seq_len) to (bs*num_tracks, seq_len)
            x_encoder_input = x.squeeze(1)
        else:
            # If input isn't mono, we cannot proceed with this encoder
            print(f"ERROR: Encoder expects MONO input, but received {num_channels} channels.")
            # Option 1: Raise an error (Safer - forces data correction)
            raise ValueError(f"Encoder input error: Expected MONO (1 channel), got {num_channels}")
            # Option 2: Try taking only the first channel (Potential data loss / hides issues)
            # print(f"WARNING: Taking only the first channel for MONO encoder due to unexpected input shape.")
            # x_encoder_input = x[:, 0, :] # Select first channel -> shape (bs*num_tracks, seq_len)
        # --- End Encoder Input Prep ---

        # generate single embedding for each track
        print(f"DEBUG: Calling self.encoder with input shape: {x_encoder_input.shape}")
        e = self.encoder(x_encoder_input) # Pass the MONO tensor
        e = e.view(bs, num_tracks, -1)  # Reshape back to (bs, num_tracks, d_embed)

        # generate the "context" embedding (logic seems okay, uses reshaped 'e')
        c = []
        for bidx in range(bs):
            # Mask applied to num_tracks dimension
            active_track_embeddings = e[bidx, ~track_mask[bidx, :], :]
            # Handle case where NO tracks are active (avoid NaN from empty mean)
            if active_track_embeddings.shape[0] > 0:
                c_n = active_track_embeddings.mean(dim=0, keepdim=True)
            else:
                print("WARNING: No active tracks for context embedding, using zeros.")
                c_n = torch.zeros(1, e.shape[-1], device=e.device, dtype=e.dtype)
            c_n = c_n.repeat(num_tracks, 1)
            c.append(c_n)
        c = torch.stack(c, dim=0)

        # fuse the track embs and context embs (logic okay)
        ec = torch.cat((e, c), dim=-1)

        # estimate mixing parameters (logic okay)
        p = self.post_processor(ec) # (bs, num_tracks, num_params)

        # generate the stereo mix
        # Reshape x back to (bs, num_tracks, channels, seq_len) before passing to mixer
        # Mixer likely expects MONO input tracks based on its internal processing
        # x = x.view(bs, num_tracks, -1) # Original, incorrect for 4D
        x_mixer_input = x.view(bs, num_tracks, num_channels, seq_len)
        if num_channels > 1:
            print("WARNING: Mixer input has multiple channels, using only the first channel.")
            x_mixer_input = x_mixer_input[:, :, 0, :] # Select first channel -> (bs, num_tracks, seq_len)
        elif num_channels == 1:
             x_mixer_input = x_mixer_input.squeeze(2) # Remove channel dim -> (bs, num_tracks, seq_len)


        # Call the mixer (mixer.forward should return y, p)
        print("DEBUG: Calling self.mixer(x_mixer_input, p)...")
        mixer_return_value = self.mixer(x_mixer_input, p) # Pass MONO tracks

        # --- Add explicit unpacking + checks ---
        print(f"DEBUG: Type of self.mixer() return value: {type(mixer_return_value)}")
        if isinstance(mixer_return_value, (list, tuple)):
            print(f"DEBUG: Length of self.mixer() return value: {len(mixer_return_value)}")
            if len(mixer_return_value) == 2:
                y, p_returned = mixer_return_value # Unpack the two values
                print("DEBUG: Mixer returned 2 values as expected.")
            else:
                print(f"ERROR: Mixer returned {len(mixer_return_value)} values, expected 2.")
                raise ValueError("Mixer forward method returned unexpected number of values.")
        else:
             print(f"ERROR: Mixer return value is not tuple/list: {type(mixer_return_value)}")
             raise TypeError("Mixer forward method returned unexpected type.")
        # --- End unpacking checks ---

        # Return the mixed audio and the NORMALIZED parameters
        print(f"DEBUG: DMC returning type(y)={type(y)}, type(p)={type(p_returned)}")
        return y, p_returned




class System(pl.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False  # Enable manual optimization

        # create the model
        # if self.hparams.automix_model == "mixwaveunet":
        #     self.model = MixWaveUNet(self.hparams.max_num_tracks, 2)
        if self.hparams.automix_model == "dmc":
            self.model = DifferentiableMixingConsole(
                self.hparams.sample_rate,
                load_weights=self.hparams.pretrained_encoder,
            )
        else:
            raise RuntimeError(f"Invalid automix_model: {self.hparams.automix_model}")

        self.recon_losses = torch.nn.ModuleDict()
        for recon_loss in self.hparams.recon_losses:
            if recon_loss == "mrstft":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "l1":
                self.recon_losses[recon_loss] = torch.nn.L1Loss()
            elif recon_loss == "sisdr":
                self.recon_losses[recon_loss] = auraloss.time.SISDRLoss()
            elif recon_loss == "sd":
                self.recon_losses[recon_loss] = auraloss.freq.SumAndDifferenceSTFTLoss(
                    fft_sizes=[4096, 1024, 256],
                    hop_sizes=[2048, 512, 128],
                    win_lengths=[4096, 1024, 256],
                )
            else:
                raise RuntimeError(f"Invalid reconstruction loss: {recon_loss}")

        self.sisdr = auraloss.time.SISDRLoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 2048, 8192],
            hop_sizes=[256, 1024, 4096],
            win_lengths=[512, 2048, 8192],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )

    def forward(self, x: torch.Tensor, track_mask: torch.Tensor = None) -> torch.Tensor:
        """Apply model to audio waveform tracks.
        Args:
            x (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
        Returns:
            y_hat (torch.Tensor): Stereo mix with shape (bs, 2, seq_len)
        """
        return self.model(x, track_mask)

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        train: bool = False,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing tracks (x) mix audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
            train (bool): Wether step is called during training (True) or validation (False).
        """
        x, y, track_mask = batch  # tracks, mix, mask

        # process input audio with model
        y_hat, params = self(x, track_mask)

        # compute loss
        loss = 0

        # compute loss on the waveform
        for loss_idx, (loss_name, loss_fn) in enumerate(self.recon_losses.items()):
            recon_loss = loss_fn(y_hat, y)
            loss += self.hparams.recon_loss_weights[loss_idx] * recon_loss

            self.log(
                ("train" if train else "val") + f"/{loss_name}",
                recon_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        sisdr_error = -self.sisdr(y_hat, y)
        # log the SI-SDR error
        self.log(
            ("train" if train else "val") + "/si-sdr",
            sisdr_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        mrstft_error = self.mrstft(y_hat, y)
        # log the MR-STFT error
        self.log(
            ("train" if train else "val") + "/mrstft",
            mrstft_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # log the overall loss
        self.log(
            ("train" if train else "val") + "/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # for plotting down the line
        data_dict = {
            "x": x.detach().float().cpu(),
            "y": y.detach().float().cpu(),
            "y_hat": y_hat.detach().float().cpu(),
            "p": params.detach().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, train=True)

        # Manually perform optimization step
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)

        if batch_idx == 0:
            return data_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        if self.hparams.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
        elif self.hparams.schedule == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                [
                    int(self.hparams.max_epochs * 0.85),
                    int(self.hparams.max_epochs * 0.95),
                ],
            )
        else:
            return optimizer

        lr_schedulers = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], lr_schedulers

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        # --- Loss functions  ---
        parser.add_argument("--recon_losses", nargs="+", default=["sd"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[1.0])
        parser.add_argument("--schedule", type=str, default="cosine")
        # --- Model ---
        parser.add_argument("--automix_model", type=str, default="dmc")
        parser.add_argument("--pretrained_encoder", action="store_true")

        return parser