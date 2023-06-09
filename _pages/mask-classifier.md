---
title: Mask Classifier
layout: post
---

<html>
<head>
	<title> Mask Classifier </title>
	<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
	<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface"></script>
</head>
<body>
	<div style="text-align: center;">
			<span id="emoji" STYLE="font-size:20pt;display:inline-block;padding-bottom:2%;">😃 Reload then wait!</span>
			<video id="video" style="margin:auto;display:block;"></video>
			<canvas id="output" style="margin:auto;position:relative;top:-480px;left:10px;"></canvas>
    </div>
</body>
<script>
	var facefind, mask_model, ctx, videoWidth, videoHeight, canvas;
	state = {
	  backend: 'webgl'
	};
	var mn=0,mo=0;
	async function setupCamera() {
		stream = await navigator.mediaDevices.getUserMedia({
			'audio': false,
		    'video': { facingMode: 'user' },
		});
		video.srcObject = stream;
	    return new Promise((resolve) => {
			video.onloadedmetadata = () => {
				resolve(video);
		    };
		});
		t = model.predict(tf.zeros([1,224,224,3]));
	}
	renderPrediction = async () => {
		tf.engine().startScope()
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.beginPath();
		predictions = await facefind.estimateFaces(video, true,false,false);
		offset = tf.scalar(127.5);
		if (predictions.length > 0) {		
		    for (let i = 0; i < predictions.length; i++) {
			    var start = predictions[i].topLeft.arraySync();
			    var end = predictions[i].bottomRight.arraySync();
			    var size = [end[0] - start[0], end[1] - start[1]];
			    if(videoWidth<end[0] || videoHeight<end[1] || start[0]<0 || start[1]<0){
			    	break
			    }
			    var inputImage = tf.browser.fromPixels(video).toFloat();
			    result= Array.from(inputImage);
				inputImage=inputImage.slice([parseInt(start[1]),parseInt(start[0]),0],[parseInt(size[1]),parseInt(size[0]),3]);
				inputImage=inputImage.resizeBilinear([224,224]).reshape([1,224,224,3]);
			    result=mask_model.predict(inputImage).dataSync()
				// result[0] result[1]
				// mask	on	mask off
				if (result[0]==1) {
					ctx.strokeStyle = "#3c784c";
					if (mo+mn>15)
						if (mn==0)
							mo--;
						else
							mn--;
					mo++;
				}
				else {
					// no mask
					ctx.strokeStyle = "#8c3535";
					if (mo+mn>15)
						if (mo==0)
							mn--;
						else
							mo--;
					mn++;
				}
				if (mo+mn>15){
					if (mo>mn) {
						document.getElementById("emoji").textContent="😷 You are wearing a mask!";
					} else {
						document.getElementById("emoji").textContent="😃 You are not wearing a mask!";
					}
				}
			    ctx.beginPath();
		        ctx.lineWidth = "4"
			    ctx.strokeRect(start[0], start[1], size[0], size[1]);
		    }     
		}
		//update frame
		requestAnimationFrame(renderPrediction);
		tf.engine().endScope()
	};

	setupPage = async () => {
	    await tf.setBackend(state.backend);
	    await setupCamera();
	    video.play();
	    videoWidth = video.videoWidth;
	    videoHeight = video.videoHeight;
	    video.width = videoWidth;
	    video.height = videoHeight;

	    canvas = document.getElementById('output');
	    canvas.width = videoWidth;
	    canvas.height = videoHeight;
	    ctx = canvas.getContext('2d');
	    ctx.fillStyle = "rgba(255, 0, 255, 1)"; 

	    facefind = await blazeface.load();
	    mask_model = await tf.loadLayersModel('https://zaforf.github.io/isp/assets/model/model.json');
		renderPrediction();
	};

	setupPage();

</script>
</html>