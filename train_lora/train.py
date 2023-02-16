import json
import math
import os
import subprocess
import pathlib
import shutil
from blip_captioning import caption_images


def train_model(
        pretrained_model_name_or_path,
        v2,
        v_parameterization,
        logging_dir,
        train_data_dir,
        reg_data_dir,
        output_dir,
        max_resolution,
        learning_rate,
        lr_scheduler,
        lr_warmup,
        train_batch_size,
        epoch,
        save_every_n_epochs,
        mixed_precision,
        save_precision,
        seed,
        num_cpu_threads_per_process,
        cache_latents,
        caption_extension,
        enable_bucket,
        gradient_checkpointing,
        full_fp16,
        no_token_padding,
        stop_text_encoder_training_pct,
        use_8bit_adam,
        xformers,
        save_model_as,
        shuffle_caption,
        save_state,
        resume,
        prior_loss_weight,
        text_encoder_lr,
        unet_lr,
        network_dim,
        lora_network_weights,
        color_aug,
        flip_aug,
        clip_skip,
        gradient_accumulation_steps,
        mem_eff_attn,
        output_name,
        model_list,  # Keep this. Yes, it is unused here but required given the common list used
        max_token_length,
        max_train_epochs,
        max_data_loader_n_workers,
        network_alpha,
        training_comment,
        keep_tokens,
        lr_scheduler_num_cycles,
        lr_scheduler_power,
        persistent_data_loader_workers,
        bucket_no_upscale,
        random_crop,
        bucket_reso_steps,
        caption_dropout_every_n_epochs, caption_dropout_rate,
):
    print('Starting Training...')
    if pretrained_model_name_or_path == '':
        print('Source model information is missing')
        return

    if train_data_dir == '':
        print('Image folder path is missing')
        return

    if not os.path.exists(train_data_dir):
        print('Image folder does not exist')
        return

    if reg_data_dir != '':
        if not os.path.exists(reg_data_dir):
            print('Regularisation folder does not exist')
            return

    if output_dir == '':
        print('Output folder path is missing')
        return

    if int(bucket_reso_steps) < 1:
        print('Bucket resolution steps need to be greater than 0')
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        print(
            'Output "stop text encoder training" is not yet supported. Ignoring'
        )
        stop_text_encoder_training_pct = 0

    # If string is empty set string to 0.
    if text_encoder_lr == '':
        text_encoder_lr = 0
    if unet_lr == '':
        unet_lr = 0

    # if (float(text_encoder_lr) == 0) and (float(unet_lr) == 0):
    #     print(
    #         'At least one Learning Rate value for "Text encoder" or "Unet" need to be provided'
    #     )
    #     return

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        # Extract the number of repeats from the folder name
        repeats = int(folder.split('_')[0])

        # Count the number of images in the folder
        num_images = len(
            [
                f
                for f in os.listdir(os.path.join(train_data_dir, folder))
                if f.endswith('.jpg')
                or f.endswith('.jpeg')
                or f.endswith('.png')
                or f.endswith('.webp')
            ]
        )

        # Calculate the total number of steps for this folder
        steps = repeats * num_images
        total_steps += steps

        # Print the result
        print(f'Folder {folder}: {steps} steps')

    # calculate max_train_steps
    max_train_steps = int(
        math.ceil(
            float(total_steps)
            / int(train_batch_size)
            * int(epoch)
            # * int(reg_factor)
        )
    )
    print(f'max_train_steps = {max_train_steps}')

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    print(f'stop_text_encoder_training = {stop_text_encoder_training}')

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    print(f'lr_warmup_steps = {lr_warmup_steps}')

    run_cmd = f'accelerate launch --num_cpu_threads_per_process={num_cpu_threads_per_process} "../train_network.py"'

    # run_cmd += f' --caption_dropout_rate="0.1" --caption_dropout_every_n_epochs=1'   # --random_crop'

    if v2:
        run_cmd += ' --v2'
    if v_parameterization:
        run_cmd += ' --v_parameterization'
    if enable_bucket:
        run_cmd += ' --enable_bucket'
    if no_token_padding:
        run_cmd += ' --no_token_padding'
    run_cmd += (
        f' --pretrained_model_name_or_path="{pretrained_model_name_or_path}"'
    )
    run_cmd += f' --train_data_dir="{train_data_dir}"'
    if len(reg_data_dir):
        run_cmd += f' --reg_data_dir="{reg_data_dir}"'
    run_cmd += f' --resolution={max_resolution}'
    run_cmd += f' --output_dir="{output_dir}"'
    run_cmd += f' --logging_dir="{logging_dir}"'
    run_cmd += f' --network_alpha="{network_alpha}"'
    if not training_comment == '':
        run_cmd += f' --training_comment="{training_comment}"'
    if not stop_text_encoder_training == 0:
        run_cmd += (
            f' --stop_text_encoder_training={stop_text_encoder_training}'
        )
    if not save_model_as == 'same as source model':
        run_cmd += f' --save_model_as={save_model_as}'
    if not float(prior_loss_weight) == 1.0:
        run_cmd += f' --prior_loss_weight={prior_loss_weight}'
    run_cmd += f' --network_module=networks.lora'

    if not (float(text_encoder_lr) == 0) or not (float(unet_lr) == 0):
        if not (float(text_encoder_lr) == 0) and not (float(unet_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --unet_lr={unet_lr}'
        elif not (float(text_encoder_lr) == 0):
            run_cmd += f' --text_encoder_lr={text_encoder_lr}'
            run_cmd += f' --network_train_text_encoder_only'
        else:
            run_cmd += f' --unet_lr={unet_lr}'
            run_cmd += f' --network_train_unet_only'
    else:
        if float(text_encoder_lr) == 0:
            print('Please input learning rate values.')
            return

    run_cmd += f' --network_dim={network_dim}'

    if not lora_network_weights == '':
        run_cmd += f' --network_weights="{lora_network_weights}"'
    if int(gradient_accumulation_steps) > 1:
        run_cmd += f' --gradient_accumulation_steps={int(gradient_accumulation_steps)}'
    if not output_name == '':
        run_cmd += f' --output_name="{output_name}"'
    if not lr_scheduler_num_cycles == '':
        run_cmd += f' --lr_scheduler_num_cycles="{lr_scheduler_num_cycles}"'
    else:
        run_cmd += f' --lr_scheduler_num_cycles="{epoch}"'
    if not lr_scheduler_power == '':
        run_cmd += f' --lr_scheduler_power="{lr_scheduler_power}"'

    run_cmd += run_cmd_training(
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_epochs=save_every_n_epochs,
        mixed_precision=mixed_precision,
        save_precision=save_precision,
        seed=seed,
        caption_extension=caption_extension,
        cache_latents=cache_latents,
    )

    run_cmd += run_cmd_advanced_training(
        max_train_epochs=max_train_epochs,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_token_length=max_token_length,
        resume=resume,
        save_state=save_state,
        mem_eff_attn=mem_eff_attn,
        clip_skip=clip_skip,
        flip_aug=flip_aug,
        color_aug=color_aug,
        shuffle_caption=shuffle_caption,
        gradient_checkpointing=gradient_checkpointing,
        full_fp16=full_fp16,
        xformers=xformers,
        use_8bit_adam=use_8bit_adam,
        keep_tokens=keep_tokens,
        persistent_data_loader_workers=persistent_data_loader_workers,
        bucket_no_upscale=bucket_no_upscale,
        random_crop=random_crop,
        bucket_reso_steps=bucket_reso_steps,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
    )

    print(run_cmd)

    subprocess.run(run_cmd, shell=True)

    # check if output_dir/last is a folder... therefore it is a diffuser model
    last_dir = pathlib.Path(f'{output_dir}/{output_name}')

    if not last_dir.is_dir():
        # Copy inference model for v2 if required
        save_inference_file(output_dir, v2, v_parameterization, output_name)


def run_cmd_training(**kwargs):
    options = [
        f' --learning_rate="{kwargs.get("learning_rate", "")}"'
        if kwargs.get('learning_rate')
        else '',
        f' --lr_scheduler="{kwargs.get("lr_scheduler", "")}"'
        if kwargs.get('lr_scheduler')
        else '',
        f' --lr_warmup_steps="{kwargs.get("lr_warmup_steps", "")}"'
        if kwargs.get('lr_warmup_steps')
        else '',
        f' --train_batch_size="{kwargs.get("train_batch_size", "")}"'
        if kwargs.get('train_batch_size')
        else '',
        f' --max_train_steps="{kwargs.get("max_train_steps", "")}"'
        if kwargs.get('max_train_steps')
        else '',
        f' --save_every_n_epochs="{kwargs.get("save_every_n_epochs", "")}"'
        if kwargs.get('save_every_n_epochs')
        else '',
        f' --mixed_precision="{kwargs.get("mixed_precision", "")}"'
        if kwargs.get('mixed_precision')
        else '',
        f' --save_precision="{kwargs.get("save_precision", "")}"'
        if kwargs.get('save_precision')
        else '',
        f' --seed="{kwargs.get("seed", "")}"' if kwargs.get('seed') else '',
        f' --caption_extension="{kwargs.get("caption_extension", "")}"'
        if kwargs.get('caption_extension')
        else '',
        ' --cache_latents' if kwargs.get('cache_latents') else '',
    ]
    run_cmd = ''.join(options)
    return run_cmd


def run_cmd_advanced_training(**kwargs):
    options = [
        f' --max_train_epochs="{kwargs.get("max_train_epochs", "")}"'
        if kwargs.get('max_train_epochs')
        else '',
        f' --max_data_loader_n_workers="{kwargs.get("max_data_loader_n_workers", "")}"'
        if kwargs.get('max_data_loader_n_workers')
        else '',
        f' --max_token_length={kwargs.get("max_token_length", "")}'
        if int(kwargs.get('max_token_length', 75)) > 75
        else '',
        f' --clip_skip={kwargs.get("clip_skip", "")}'
        if int(kwargs.get('clip_skip', 1)) > 1
        else '',
        f' --resume="{kwargs.get("resume", "")}"'
        if kwargs.get('resume')
        else '',
        f' --keep_tokens="{kwargs.get("keep_tokens", "")}"'
        if int(kwargs.get('keep_tokens', 0)) > 0
        else '',
        f' --caption_dropout_every_n_epochs="{kwargs.get("caption_dropout_every_n_epochs", "")}"'
        if int(kwargs.get('caption_dropout_every_n_epochs', 0)) > 0
        else '',
        f' --caption_dropout_rate="{kwargs.get("caption_dropout_rate", "")}"'
        if float(kwargs.get('caption_dropout_rate', 0)) > 0
        else '',

        f' --bucket_reso_steps={int(kwargs.get("bucket_reso_steps", 1))}'
        if int(kwargs.get('bucket_reso_steps', 64)) >= 1
        else '',

        ' --save_state' if kwargs.get('save_state') else '',
        ' --mem_eff_attn' if kwargs.get('mem_eff_attn') else '',
        ' --color_aug' if kwargs.get('color_aug') else '',
        ' --flip_aug' if kwargs.get('flip_aug') else '',
        ' --shuffle_caption' if kwargs.get('shuffle_caption') else '',
        ' --gradient_checkpointing'
        if kwargs.get('gradient_checkpointing')
        else '',
        ' --full_fp16' if kwargs.get('full_fp16') else '',
        ' --xformers' if kwargs.get('xformers') else '',
        ' --use_8bit_adam' if kwargs.get('use_8bit_adam') else '',
        ' --persistent_data_loader_workers'
        if kwargs.get('persistent_data_loader_workers')
        else '',
        ' --bucket_no_upscale' if kwargs.get('bucket_no_upscale') else '',
        ' --random_crop' if kwargs.get('random_crop') else '',
    ]
    run_cmd = ''.join(options)
    return run_cmd


def save_inference_file(output_dir, v2, v_parameterization, output_name):
    # List all files in the directory
    files = os.listdir(output_dir)

    # Iterate over the list of files
    for file in files:
        # Check if the file starts with the value of output_name
        if file.startswith(output_name):
            # Check if it is a file or a directory
            if os.path.isfile(os.path.join(output_dir, file)):
                # Split the file name and extension
                file_name, ext = os.path.splitext(file)

                # Copy the v2-inference-v.yaml file to the current file, with a .yaml extension
                if v2 and v_parameterization:
                    print(
                        f'Saving v2-inference-v.yaml as {output_dir}/{file_name}.yaml'
                    )
                    shutil.copy(
                        f'./v2_inference/v2-inference-v.yaml',
                        f'{output_dir}/{file_name}.yaml',
                    )
                elif v2:
                    print(
                        f'Saving v2-inference.yaml as {output_dir}/{file_name}.yaml'
                    )
                    shutil.copy(
                        f'./v2_inference/v2-inference.yaml',
                        f'{output_dir}/{file_name}.yaml',
                    )


def main():
    config_file_path = 'default_config.json'

    with open(config_file_path, 'r') as config_file:
        print('Loading config...')
        config = json.load(config_file)

    config['train_data_dir'] = os.path.expanduser(config['train_data_dir'])
    config['output_dir'] = os.path.expanduser(config['output_dir'])
    config['reg_data_dir'] = os.path.expanduser(config['reg_data_dir'])
    config['logging_dir'] = os.path.expanduser(config['logging_dir'])

    caption_images(
        train_data_dir=config['train_data_dir'], prefix=config['output_name'])
    train_model(**config)


if __name__ == "__main__":
    main()
