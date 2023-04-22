docker run --rm -it \
			--net=host \
			--name=fake_jphw \
			-v `pwd`/.:/hopny \
			quhu_gpu:2.2.0 bash