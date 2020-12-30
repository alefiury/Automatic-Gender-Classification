mkdir -p data

if [ ! -d test-clean ]; then
    pushd data

    echo 'DOWNLOADING DATASET...'
    wget -c https://www.openslr.org/resources/12/test-clean.tar.gz

    mkdir -p test-clean

    echo 'UNZIPPING FILE...'
    tar -C test-clean -xvzf test-clean.tar.gz

    popd

    python utils/converter.py -c --files_path='./data/test-clean' --in_extension='.flac'
    python utils/converter.py -m --files_path='./data/test-clean' --dataset_type='test-clean' --folder_name='test'
    rm -r -f data/test-clean
fi

if [ ! -d dev-clean ]; then
    pushd data

    echo 'DOWNLOADING DATASET...'
    wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz

    mkdir -p dev-clean

    echo 'UNZIPPING FILE...'
    tar -C dev-clean -xvzf dev-clean.tar.gz

    popd

    python utils/converter.py -c --files_path='./data/dev-clean' --in_extension='.flac'
    python utils/converter.py -m --files_path='./data/dev-clean' --dataset_type='dev-clean' --folder_name='dev'
    rm -r -f data/dev-clean
fi

if [ ! -d train-clean-100 ]; then
    pushd data

    echo 'DOWNLOADING DATASET...'
    wget -c https://www.openslr.org/resources/12/train-clean-100.tar.gz

    mkdir -p train-clean

    echo 'UNZIPPING FILE...'
    tar -C train-clean -xvzf train-clean-100.tar.gz

    popd

    python utils/converter.py -c --files_path='./data/train-clean' --in_extension='.flac'
    python utils/converter.py -m --files_path='./data/train-clean' --dataset_type='train-clean-100' --folder_name='train'
    rm -r -f data/train-clean
fi

pushd utils
wget -c https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt
popd

    