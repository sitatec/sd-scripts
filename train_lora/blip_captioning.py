import os
import subprocess


def caption_images(
    train_data_dir,
    prefix = '',
    postfix = '',
    caption_file_ext = '.txt',
    batch_size = 1,
    max_length = 75,
    min_length = 5,
    top_p = 0.9,
    beam_search = True,
    num_beams = 1,
):
    if train_data_dir == '':
        print('Image folder is missing')
        return

    if caption_file_ext == '':
        print('Please provide an extension for the caption files.')
        return

    print(f'Captioning files in {train_data_dir}...')
    run_cmd = f'python ../finetune/make_captions.py'
    run_cmd += f' --batch_size="{int(batch_size)}"'
    run_cmd += f' --num_beams="{int(num_beams)}"'
    run_cmd += f' --top_p="{top_p}"'
    run_cmd += f' --max_length="{int(max_length)}"'
    run_cmd += f' --min_length="{int(min_length)}"'
    if beam_search:
        run_cmd += f' --beam_search'
    if caption_file_ext != '':
        run_cmd += f' --caption_extension="{caption_file_ext}"'
    run_cmd += f' "{train_data_dir}"'
    run_cmd += f' --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"'

    print(run_cmd)

    subprocess.run(run_cmd, shell=True)

    # Add prefix and postfix
    add_pre_postfix(
        folder=train_data_dir,
        caption_file_ext=caption_file_ext,
        prefix=prefix,
        postfix=postfix,
    )

    print('Captioning done')

def has_ext_files(directory, extension):
    # Iterate through all the files in the directory
    for file in os.listdir(directory):
        # If the file name ends with extension, return True
        if file.endswith(extension):
            return True
    # If no extension files were found, return False
    return False

def add_pre_postfix(
    folder='', prefix='', postfix='', caption_file_ext='.txt'
):
    if not has_ext_files(folder, caption_file_ext):
        print(
            f'No files with extension {caption_file_ext} were found in {folder}...'
        )
        return

    if prefix == '' and postfix == '':
        return

    files = [f for f in os.listdir(folder) if f.endswith(caption_file_ext)]
    if not prefix == '':
        prefix = f'{prefix} '
    if not postfix == '':
        postfix = f' {postfix}'

    for file in files:
        with open(os.path.join(folder, file), 'r+') as f:
            content = f.read()
            content = content.rstrip()
            f.seek(0, 0)
            f.write(f'{prefix}{content}{postfix}')
    f.close()
