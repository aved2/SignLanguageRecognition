// Import dependencies
import React, { useRef, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
//import { nextFrame } from "@tensorflow/tfjs";
// 2. TODO - Import drawing utility here
// e.g. import { drawRect } from "./utilities";
import {drawRect} from "./utilities"; 


function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Main function
  const runCoco = useCallback(async () => {
    // 3. TODO - Load network 
    // e.g. const net = await cocossd.load();
    // https://cloud-object-storage-cos-standard-vnm.s3.us-east.cloud-object-storage.appdomain.cloud/model.json

    console.log("runCoco is called");

    try {
      // Load model
      console.log("Model URL:", process.env.REACT_APP_MODEL_URL);
      const net = await tf.loadGraphModel(process.env.REACT_APP_MODEL_URL);
      console.log("Model loaded");

      // Loop and detect hands
      setInterval(() => {
        detect(net);
      }, 16.7);
    } catch (error) {
      console.error("Error loading model: ", error);
    }
  }, []);



  const detect = async (net) => {
    console.log("detect is called");  
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video)
      const resized = tf.image.resizeBilinear(img, [640,480])
      const casted = resized.cast('int32')
      const expanded = casted.expandDims(0)
      const obj = await net.executeAsync(expanded)
      console.log(obj)

      const boxes = await obj[1].array()
      const classes = await obj[2].array()
      const scores = await obj[4].array()
      
      // Draw mesh
      const ctx = canvasRef.current.getContext("2d");

      // 5. TODO - Update drawing utility
      // drawSomething(obj, ctx)  
      requestAnimationFrame(()=>{drawRect(boxes[0], classes[0], scores[0], 0.8, videoWidth, videoHeight, ctx)}); 
      tf.dispose(img)
      tf.dispose(resized)
      tf.dispose(casted)
      tf.dispose(expanded)
      tf.dispose(obj)

    }
  };

  useEffect(()=>{runCoco()},[runCoco]);

  return (
    <div className="App">
      <header className="App-header">
      <div className="header">
        <h1 className="title">Real Time Sign Lanuage Detection</h1>
      </div>
      <div className="instructBox">
        <p className="instruct">For best results, use a very well lit room with a blank background</p>
        <p className="instruct">Position only your hand in the frame, 1-2 feet from the camera</p>
        <p className="instruct">Allow browser camera permissions, test model with:</p>
        
        <div className="imgs">
        <div className="hands">
          <p className="gests">Hello</p>
          <img className="img" src="/images/hello.png" alt="Hello in ASL" />
        </div>
        <div className="hands">
        <p className="gests">Yes</p>
        <img className="img" src="/images/yes.png" alt="Hello in ASL" />
        </div>
        <div className="hands">
        <p className="gests">No</p>
        <img className="img" src="/images/no.png" alt="Hello in ASL" />
        </div>
        <div className="hands">
        <p className="gests">I Love You</p>
        <img className="img" src="/images/ily.png" alt="Hello in ASL" />
        </div>
        <div className="hands">
        <p className="gests">Thank You</p>
        <img className="img" src="/images/ty.png" alt="Hello in ASL" />
        </div>
        </div>
        <p className=" footer ">*Model is in initial training stage. Currently is only trained on the given five poses, working on expanding training to improve accuracy.</p>
        </div>
        <Webcam
        className= "frame"
          ref={webcamRef}
          muted={true} 
        />

        <canvas
          ref={canvasRef}
          className= "frame"
      
        />
      </header>
    </div>
  );
}

export default App;
