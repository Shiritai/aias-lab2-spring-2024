script_dir=$(realpath $(dirname $0))

file_name=libtorch-shared-with-deps-latest.zip
file_path=${script_dir}/${file_name}

check_and_install() {
    if ! command -v $1 > /dev/null
    then
        sudo apt update && sudo apt install $1 -y
    fi
}

prefer_pkg="wget cmake unzip"
for item in ${prefer_pkg}; do
    check_and_install "${item}"
done

cpp_path="${script_dir}/cpp"
libtorch_path="${cpp_path}/libtorch"
build_path="${cpp_path}/build"

if [ ! -f $file_path ]
then
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
fi
if [ ! -d $libtorch_path ]
then
    unzip libtorch-shared-with-deps-latest.zip  -d $cpp_path
fi

mkdir -p ${build_path}
cd ${build_path} && cmake ..