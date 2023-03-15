package com.example.bildverarbeitungsapp3;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.*;

/**
 * Funktionen zum Erstellen für das Template
 *
 * @author  Niklas Hübner
 * @version 1.0
 */
public final class TemplateDetection {
    /*
     * Schwellenwerte für den Canny-Algorithmus
     * THRESHOLD2 = 2*THRESHOLD1 oder
     * THRESHOLD2 = 3*THRESHOLD1
     */
    private static final double THRESHOLD1 = 75;
    private static final double THRESHOLD2 = 150;
    /**
     * Class constructor.
     */
    private TemplateDetection() {
    }

    /**
     * Originalbild wird um die hälfte verkleinert
     * @param src Originalbild
     * @return    Originalbild halbiert
     */
    public static Mat scaleMat(Mat src) {

        // Creating an empty matrix to store the result
        Mat dst = new Mat();
        // Creating the Size object
        //Size size = new Size(src.cols() * 0.5, src.rows() * 0.5);
        Size size = new Size(150,150);
        // Scaling the Image
        Imgproc.resize(src, dst, size, 0, 0, Imgproc.INTER_AREA);
        System.out.println("rescaled");
        return dst;
    }

    /**
     * Umwandeln in ein Graustufenbild
     * @param src Originalbild
     * @return    Graustufenbild
     */
    public static Mat colorToGray(Mat src) {
        // convert into gray image
        Mat grayMat = new Mat();
        Imgproc.cvtColor(src, grayMat, Imgproc.COLOR_BGR2GRAY);
        Log.d("COLTOGRAY", "color to gray");
        return grayMat;
    }

    /**
     * Rect edgeDetection(Mat src)
     *
     * Zusammenfassung der edgeDetection Funktion:
     *
     *    Sucht das Rechteck welches am besten für das Template geeignet ist
     *
     * @param src Mat Originalbild
     * @return    Rechteck
     */
    public static Rect edgeDetection(Mat src) throws IOException {

        /**
         * Umwandel in ein Graustufenbild
         */
        Mat gray = colorToGray(src);

        /**
         * Gaussian Blur
         * Kernel von 3x3 funktioniert für die Tests gut, mit sigmaX Wert von 1
         */
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(3, 3), 1);
        Imgcodecs.imwrite("res/drawable\n" +
                "res/drawable-v24/blur.jpg", blur);

        /**
         * Kantenbild
         * Canny threshold Werte: sollten im Verhältnis von 1:2 oder 1:3 sein (lower:upper)
         * Kernel Size von 3, siehe Dokumentation
         */
        Mat edges = new Mat();
        Imgproc.Canny(blur, edges, THRESHOLD1, THRESHOLD2,3);
        Imgcodecs.imwrite("Bilder/canny.jpg", edges);


        /**
         * Kantenfinden
         * RETR_EXTERNAL wird verwendet, da es die Außenkontur gibt.
         * CHAIN_APPROX_SIMPLE wird verwendet, spart viel Speicher.
         */
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        //Imgcodecs.imwrite("Bilder/drawcontours3.jpg", edges);
        System.out.println("Hierarch elementsize: " + hierarchy.elemSize());

        /**
         * Zeichne die Konturen
         * contourIdx: -1, damit alle Konturen eingezeichnet werden
         */
        Mat contourMat = Mat.zeros(gray.size(), CvType.CV_8UC3);
        Scalar white = new Scalar(255, 0, 255);
        Imgproc.drawContours(contourMat, contours, -1, white);
        Imgcodecs.imwrite("Bilder/drawcontours1.jpg", contourMat);

        /**
         * Füllen von Polygon
         */
        for (MatOfPoint contour : contours) {
            Imgproc.fillPoly(contourMat, Collections.singletonList(contour), white);
        }
        Imgcodecs.imwrite("Bilder/filled2.jpg", contourMat);

        /*
        Mat kernel = Mat.ones(5,5, CvType.CV_32F);
        Imgproc.morphologyEx(contourMat, contourMat, Imgproc.MORPH_OPEN, kernel);
        Imgcodecs.imwrite("Bilder/morphOpen.jpg", contourMat);
        */

        /**
         * Rechtecke um die gefundenen Konturen einzeichnen.
         */
        Scalar green = new Scalar(81, 180, 0);
        List<Rect> rectList = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(new MatOfPoint2f(contour.toArray()));
            rectList.add(rect);
            Imgproc.rectangle(contourMat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), green, 2);
        }
        Imgcodecs.imwrite("Bilder/rect3.jpg", contourMat);


        Rect area = averageArea(rectList);
        System.out.println("Area: " + area.area());

        Imgcodecs.imwrite("Bilder/contours4.jpg", contourMat);
        Log.d("EDGEDET", "success");
        return area;
    }


    /**
     * Schneidet das Template aus dem Originalbild aus.
     *
     * @param image_original Originalbild
     * @param rect           Rechteck welches zuvor in edgeDetection bestimmt wurde.
     * @return               Template welches für TemplateMatching und später für Tensorflow benötigt wird.
     */
    public static Mat cropTemplate(Mat image_original, Rect rect) {

        Mat image_output = image_original.submat(rect);
        Imgcodecs.imwrite("Bilder/image_output2.jpg", image_output);
        Imgcodecs.imwrite("Bilder/image_recangle2.jpg", image_original);
        Log.d("CROP", "success");
        return image_output;
    }

    /**
     * Entfernt alle Flächen mit dem Wert 0.
     *
     * @param rectList Liste von allen gefundenen Rechtecken
     * @return         Liste ohne Werte, die Fläche 0 haben
     */
    public static List<Rect> removeZeros(List<Rect> rectList) {
        Log.d("REMZERO", "success");
        rectList.removeIf(r -> r.area() == 0);

        return rectList;
    }

    /**
     * Berechnet die Durchschnittsfläche der Rechtecke
     * Sucht das Rechteck welches am besten zum Mittelwert passt
     *
     * @param rectList Liste von allen gefundenen Rechtecken
     * @return         Rechteck von welchem die Flache am besten zum Mittelwert passt
     */
    public static Rect averageArea(List<Rect> rectList) {
        Log.d("AVRARES", "success");
        double sum = 0;

        rectList = removeZeros(rectList);

        /*
         * Mittelwert Berechnung
         * nach der Summe von allen Rechteckflächen wird durch die Anzahl der Rechtecke geteilt
         * zwei Möglichkeiten den Mittelwert zu bekommen
         */
        final double mean = rectList.stream().mapToDouble(Rect::area).average().orElse(-1);
        /*
        for (Rect r : rectList) {
            sum += r.area();
        }
        mean = sum / rectList.size();
        */

        /*
         * Prüfen welches Rechteck am nächsten beim Mittelwert ist
         * dieses Recheck ist der Rückgabewert dieser Funktion
         */

        /*
        double distance = Math.abs(rectList.get(0).area() - mean);
        int index = 0;
        for (int c = 1; c < rectList.size(); c++) {
            double cdistance = Math.abs(rectList.get(c).area() - mean);
            if (cdistance < distance) {
                index = c;
                distance = cdistance;
            }
        }
        double finalArea = rectList.get(index).area();
        Rect rect = rectList.get(index);
        System.out.println("Mittelwert: " + mean + " FinalArea: " + finalArea); */
        return rectList.stream().min(Comparator.comparingDouble(rect->Math.abs(rect.area()-mean))).orElse(null);
    }
}
