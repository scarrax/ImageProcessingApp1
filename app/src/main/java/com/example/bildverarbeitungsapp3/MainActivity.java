package com.example.bildverarbeitungsapp3;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContract;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.CountDownTimer;
import android.os.Handler;
import android.provider.MediaStore;
import android.view.View;
import android.util.Log;
import android.widget.Button;

import android.os.Bundle;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.example.bildverarbeitungsapp3.ml.MobilenetV110224Quant;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class MainActivity extends AppCompatActivity {
    /*
     * Schwellenwert welcher sich gut für die Tests geeignet hat
     * gut für die Test 0.6-0.85
     */
    private static final double THRESHOLD = 0.7;

    // BChooseImage from gallery
    Button BSelectFromGallery;

    // BImage from Camera
    Button BSelectFromCamera;

    // Button Predict
    Button BPredict;

    // Preview Image
    ImageView IVPreviewImage;

    private Bitmap bMap;

    // Textviews
    TextView txtAnzObj;
    TextView txtObjDec;
    int anz = 0;

    // Progressbar variable
    private ProgressBar PBLoading;
    private int progressStatus = 0;
    private Button BStartProgress;
    private CountDownTimer countDownTimer;

    ActivityResultLauncher<String> mGetContent;
    ActivityResultLauncher<Uri> mGetCameraContent;

    // input labels
    StringBuilder sb = new StringBuilder();
    InputStream is;
    BufferedReader br;
    String line;
    ArrayList<String> label = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Enable OpenCV
        if(OpenCVLoader.initDebug())Log.d("LOADED", "success");
        else Log.d("LOADED", "error");

        // register the UI with their appropriate IDs
        BSelectFromGallery = findViewById(R.id.btnPicChoose);
        IVPreviewImage = findViewById(R.id.imgViewPic);
        BSelectFromCamera = findViewById(R.id.btnPicRecord);
        BPredict = findViewById(R.id.btnPredict);

        // progressBar initializing variable with ids
        BStartProgress = findViewById(R.id.btnStart);
        BStartProgress.setEnabled(false);
        BPredict.setEnabled(false);
        PBLoading = findViewById(R.id.proBar);

        // Textview
        txtAnzObj = findViewById(R.id.txtAnzObj);
        txtObjDec = findViewById(R.id.txtGefundObj);

        PBLoading.setProgress(progressStatus);
        countDownTimer = new CountDownTimer(5000,1000) {
            @Override
            public void onTick(long millisUntilFinished) {
                Log.v("on_Tik", "Tick of Progress" + progressStatus + millisUntilFinished);
                progressStatus++;
                PBLoading.setProgress(progressStatus*100/(5000/1000));
            }

            @Override
            public void onFinish() {
                progressStatus++;
                PBLoading.setProgress(100);
            }
        };

        BStartProgress.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                txtAnzObj.setText("Anzahl der gefundenen Objekte: ");
                //IVPreviewImage.setImageResource(R.drawable.erbsen2);
                bMap = ((BitmapDrawable)IVPreviewImage.getDrawable()).getBitmap();
                //Bitmap bMap = BitmapFactory.decodeResource(getResources(), R.drawable.erbsen2);
                Mat image = new Mat();
                Utils.bitmapToMat(bMap, image);

                /*
                 * Template bekommen
                 */
                Mat template = new Mat();
                template = ImageDetection(image);

                image = TemplateDetection.scaleMat(image);
                image = TemplateDetection.colorToGray(image);
                try {
                    anz = ImageMatching(image ,template, THRESHOLD);
                    txtAnzObj.append(String.valueOf(anz));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //countDownTimer.start();
                BPredict.setEnabled(true);
            }
        });

        BPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                bMap = Bitmap.createScaledBitmap(bMap, 224,224,true);
                Log.d("PREDICT", "start");
                try {
                    loadLabels();
                    Log.d("LABELS", "success");
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.d("LABELS", "error");
                }
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(getApplicationContext());


                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.UINT8);

                    TensorImage tensorImage = new TensorImage(DataType.UINT8);
                    tensorImage.load(bMap);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();

                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.

                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    System.out.println("Data type: "+outputFeature0.getDataType().toString());

                    float[] data = outputFeature0.getFloatArray();
                    int maxIdx = 0;

                    for(int i = 0; i < data.length; i++){
                        if(data[maxIdx] < data[i]){
                            maxIdx = i;
                        }
                    }
                    txtObjDec.setText(label.get(maxIdx));
                    // Releases model resources if no longer used.
                    model.close();
                    Log.d("PREDICT", "success");
                    //txtObjDec.setText(outputFeature0.getFloatArray()[0]+ "\n" + outputFeature0.getFloatArray()[1]);

                } catch (IOException e) {
                    // TODO Handle the exception
                    Log.d("PREDICT", "error");
                }

            }
        });

        mGetContent = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                new ActivityResultCallback<Uri>() {
            @Override
            public void onActivityResult(Uri result) {

                IVPreviewImage.setImageURI(result);
                BStartProgress.setEnabled(true);
            }
        });


        mGetCameraContent = registerForActivityResult(new ActivityResultContracts.TakePicture(),
                new ActivityResultCallback<Boolean>() {
            @Override
            public void onActivityResult(Boolean result) {


            }
        });

        BSelectFromGallery.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){

                mGetContent.launch("image/*");
            }
        });

        BSelectFromCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //mGetCameraContent.launch();
            }
        });

        Log.d("onCreat", "in onCreate");
    }

    /**
     * Bild vorbereiten für TemplateDetection.
     * Aufruf edgeDetection für das Rechteck welches im nächsten Schritt benötigt wird.
     * Aufruf cropTemplate um das Template zu erhalten.
     * Template in Graustufenbild konvertieren.
     * Originalbild in Graustufenbild konvertieren.
     */
    public Mat ImageDetection(Mat image){
        Log.d("IMGDET", "success");

        image = TemplateDetection.scaleMat(image);
        Rect rect = null;
        try {
            rect = TemplateDetection.edgeDetection(image);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Mat template = TemplateDetection.cropTemplate(image, rect);
        template = TemplateDetection.colorToGray(template);
        return template;
    }

    public int ImageMatching(Mat image, Mat template, double threshold) throws IOException {

        Log.d("IMGMAT", "success");
        /**
         * Rückgabewert ist eine Liste von Points wo der maxValue >= threshold ist.
         * Als Methode wird TM_CCOOEDD_NORMED verwendet, diese funktioniert mit dem maxValue.
         */
        List<Point> detectedPoints = TemplateMatching.detectTemplate(image,template, threshold);
        /**
         * Es werden überlappende Punkte gefunden, in python wäre die Lösung groupRectangles.
         */
        List<Point> totalPoints = TemplateMatching.removeNearPoints(detectedPoints, image, template);
        return totalPoints.size();
    }

    public void loadLabels() throws IOException {
        Log.d("labels", "start");

        is = getAssets().open("labels.txt");
        br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
        while((line=br.readLine()) != null){
            label.add(line);
        }
        br.close();
        Log.d("labels", "end");
    }

}