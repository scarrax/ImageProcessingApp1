package com.example.bildverarbeitungsapp3;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Alle Funktionen für Matching Template
 *
 * @author  Niklas Hübner
 * @version 1.0
 */
public final class TemplateMatching {

    /**
     * Class constructor.
     */
    private TemplateMatching(){}

    /**
     * Finden von übereinstimmungen vom Template und dem Originalbild.
     *
     * @param srcImage      Originalbild
     * @param templateImage Template (das gesuchte Bild)
     * @param threshold     Wie genau das Template auf die Region im Originalbild passen muss
     * @return              List von Punkten wo eine Übereinstimmung von Originalbild und Template ist
     */
    public static List<Point> detectTemplate(Mat srcImage, Mat templateImage, double threshold){

        /**
         * Variablen Deklaration
         */
        List<Point> detectedPoints = new ArrayList<Point>();
        List<Double> detectedValue = new ArrayList<>();
        Point matchLoc;
        double maxvalue;
        Mat dst = srcImage.clone();

        /**
         * Deklaration der Mat resultMat
         */
        int result_cols = srcImage.cols() - templateImage.cols() + 1;
        int result_rows = srcImage.rows() - templateImage.rows() + 1;
        Mat resultMat = new Mat();
        //Mat resultMat = new Mat(result_rows, result_cols, CvType.CV_32FC1);
        resultMat.create(result_rows, result_cols, CvType.CV_32FC1);

        /**
         * Erster durchlauf um zu Prüfen, ob das Template im Originalbild ist.
         * TM_CCOEFF_NORMED liefert das beste Ergebnis beim max value.
         * matchLoc gibt den aktuellen Punkt an für das maxValue.
         * maxvalue in Range (0,1) gibt an wie gut das Template mit dem Punkt übereinstimmt.
         * Lokalisieren der minimum und maximum Werte in der Result Matrix mit Core.minMaxLoc
         */
        Imgproc.matchTemplate(dst, templateImage, resultMat, Imgproc.TM_CCOEFF_NORMED);
        Core.normalize(resultMat, resultMat, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        Core.MinMaxLocResult mmr = Core.minMaxLoc(resultMat);
        matchLoc = mmr.maxLoc;
        maxvalue = mmr.maxVal;
        //Imgcodecs.imwrite("Bilder/resultMat.jpg", resultMat);
        /**
         * Gefunden Werte den entsprechenden Listen hinzufügen.
         */
        detectedPoints.add(matchLoc);
        detectedValue.add(maxvalue);
        System.out.println("List of points " + detectedPoints);
        System.out.println("List of maxvalue " + detectedValue);


        /**
         * Einzeichnen vom Rechteck an der Location von matchLoc,
         * im Originalbild und in der result Matrix.
         */
        //Imgproc.rectangle(dst, matchLoc, new Point(matchLoc.x + templateImage.cols(),
        //        matchLoc.y + templateImage.rows()), new Scalar(0, 0, 0), 2, 8, 0);
        Imgproc.rectangle(resultMat, matchLoc, new Point(matchLoc.x + templateImage.cols(), matchLoc.y + templateImage.rows()),
                new Scalar(0, 0, 0), 2, 8, 0);
        //Imgcodecs.imwrite("Bilder/resultMat.jpg", resultMat);

        /**
         * Schleife wird so lange durchlaufen bis der maximal Wert unter dem threshold ist.
         * Der maximal Wert wird mit jedem Durchlauf kleiner.
         */
        while(true){
            mmr = Core.minMaxLoc(resultMat);
            matchLoc = mmr.maxLoc;
            maxvalue = mmr.maxVal;
            System.out.println("mmr: " + maxvalue);

            if(maxvalue >= threshold){
                detectedPoints.add(matchLoc);
                detectedValue.add(maxvalue);

                //System.out.println("Template Matches with input image");
                Imgproc.rectangle(dst, matchLoc, new Point(matchLoc.x + templateImage.cols(),matchLoc.y + templateImage.rows()),
                        new Scalar(0,255,0), 1,8,0);
                Imgproc.rectangle(resultMat, matchLoc, new Point(matchLoc.x + templateImage.cols(),matchLoc.y + templateImage.rows()),
                        new Scalar(0, 255, 0), 1,8,0);
            }else{
                break;
            }
        }

        resultMat.convertTo(resultMat, CvType.CV_8UC1, 255.0);

        String file3 = "Bilder/MultiMatchTemp2.jpg";
        Imgcodecs.imwrite(file3, dst);
        String file5 = "Bilder/result4.jpg";
        Imgcodecs.imwrite(file5, resultMat);

        Log.d("MATCH", "success");
        return detectedPoints;
    }

    /**
     * Entfernen von überlappenden Rechtecken.
     *
     * @param listPoints    Liste von Punkten, alle Punkte die über den threshold liegen.
     * @param srcImage      Originalbild
     * @param templateImage Template, um die finalen Rechtecke einzuzeichnen.
     * @return              Liste von Punkten ohne Überlappungen.
     * @throws IOException
     */
    public static List<Point> removeNearPoints(List<Point> listPoints, Mat srcImage, Mat templateImage) throws IOException {
        System.out.println("ListofPoints: " + listPoints);

        /**
         * Startpunkte
         */
        List<Point> totalPoints = new ArrayList<>();
        totalPoints.add(listPoints.get(0));
        System.out.println("Add Points: " + listPoints.get(0));
        Imgproc.rectangle(srcImage, listPoints.get(0), new Point(listPoints.get(0).x + templateImage.cols(),
                listPoints.get(0).y + templateImage.rows()), new Scalar(0, 0, 0), 1, 8, 0);


        String file5 = "Bilder/firstRectangle.jpg";
        Imgcodecs.imwrite(file5, srcImage);

        /**
         * Entfernen von zu naher Rechtecke.
         * Es wird überprüft, ob der aktuelle Punkt in einer Distanz zu einem anderen Punkt liegt.
         * Wenn inDistance true ist, ist der Wert ok und liegt außerhalb der vereinbarten Distanz
         * distance auf 20, funktioniert für die meisten Testversuche.
         */
        boolean inDistance = false;
        double distance = 20;
        for(Point p: listPoints.subList(1, listPoints.size())){
            for(Point n : totalPoints){
                if((p.x == n.x) & (p.y == n.y)){
                    continue;
                }else if (Math.hypot((p.x-n.x), (p.y-n.y)) > distance){
                    inDistance = true;
                }else{
                    inDistance = false;
                    break;
                }
            }
            if(inDistance){
                totalPoints.add(p);
                Imgproc.rectangle(srcImage, p, new Point(p.x + templateImage.cols(),
                        p.y + templateImage.rows()), new Scalar(0, 0, 0), 1, 8, 0);
                System.out.println("nicht gleiche werte: "+ p.x+ " " + p.y);
                System.out.println("totalpointssize: " + totalPoints.size());
            }

        }

        String file3 = "Bilder/AfterRemoveNearPoints2.jpg";
        Imgcodecs.imwrite(file3, srcImage);

        System.out.println("Es wurden: " + totalPoints.size() + " Objekte gefunden.");
        Log.d("REMOVEPOINTS", "success");
        return totalPoints;
    }
}

