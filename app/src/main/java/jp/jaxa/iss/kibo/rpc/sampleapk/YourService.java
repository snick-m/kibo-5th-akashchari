package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import gov.nasa.arc.astrobee.types.Point;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Quaternion;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.*;

import org.opencv.android.Utils;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.aruco.*;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import javax.vecmath.AxisAngle4f;
import javax.vecmath.Matrix3f;
import javax.vecmath.Quat4f;
import javax.vecmath.Vector3f;

/**
 * Class meant to handle commands from the Ground Data System and execute them
 * in Astrobee.
 */

class NoArucoException extends Exception {
    public NoArucoException(String str) {
        super(str);
    }
}

public class YourService extends KiboRpcService {
    private static final float THRESHOLD = 0.7f;
    private MatOfDouble camMatrix, distCoeffs;
    private ObjectDetector detector;
    private List<String> foundItems = new LinkedList<>();
    private String targetItem = "";
    private List<Mat> areaImages = new LinkedList<>();
    private List<Mat> arucoCorners = new LinkedList<>();
    private List<Quaternion> faceOrients = new LinkedList<>();
    private boolean SKIPAREA1 = false;

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelName) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private Quaternion multiplyQuaternions(Quaternion q1, Quaternion q2) {
        float w = q1.getW() * q2.getW() - q1.getX() * q2.getX() - q1.getY() * q2.getY() - q1.getZ() * q2.getZ();
        float x = q1.getX() * q2.getW() + q1.getW() * q2.getX() + q1.getY() * q2.getZ() - q1.getZ() * q2.getY();
        float y = q1.getY() * q2.getW() + q1.getW() * q2.getY() + q1.getZ() * q2.getX() - q1.getX() * q2.getZ();
        float z = q1.getZ() * q2.getW() + q1.getW() * q2.getZ() + q1.getX() * q2.getY() - q1.getY() * q2.getX();
        return new Quaternion(x, y, z, w);
    }

    private Quaternion opencvToKiboSpace(Quaternion q1) {
        Quaternion rt = new Quaternion(0f, 1f, 0f, 0f);
        return multiplyQuaternions(q1, rt);
    }

    private void initDetector() {
        Log.i("TEST", "initDetector: Start Initialization");
        String modelName = "model_fp16.tflite";
        ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(0.75f)
                .setMaxResults(25)
                .build();
        try {
            MappedByteBuffer modelByteBuffer = loadModelFile(getAssets(), modelName);
            detector = ObjectDetector.createFromBufferAndOptions(modelByteBuffer, options);
            Log.i("TEST", "initDetector: " + detector.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Mat undistort(Mat image, Integer count) {
        if (image != null) {
//            api.saveMatImage(image, "image_" + count + ".png");
            Mat undistortedImage = new Mat();
            Calib3d.undistort(image, undistortedImage, camMatrix, distCoeffs);
//            api.saveMatImage(undistortedImage, String.format("image_%da_undistorted.png", count));
            return undistortedImage;
        }
        return null;
    }

    private void detectAndSaveAruco(Mat image, Integer count) throws NoArucoException {
        if (image != null) {
            DetectorParameters params = DetectorParameters.create();
            Mat ids = new MatOfInt();
            List<Mat> corners = new LinkedList<>();
            Dictionary dict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);

            org.opencv.core.Point p1 = new org.opencv.core.Point(217.5, 13.5);
            org.opencv.core.Point p2 = new org.opencv.core.Point(267.5, 13.5);
            org.opencv.core.Point p3 = new org.opencv.core.Point(267.5, 63.5);
            org.opencv.core.Point p4 = new org.opencv.core.Point(217.5, 63.5);

            MatOfPoint2f dest = new MatOfPoint2f(p1, p2, p3);

            Mat drawMarkers = image.clone();

            Aruco.detectMarkers(drawMarkers, dict, corners, ids, params);
//            Aruco.drawDetectedMarkers(drawMarkers, corners, ids);
//            api.saveMatImage(drawMarkers, "image_" + count + "b_detected.png");
            if (corners.size() > 0) {
                Mat cornersMat = corners.get(0);
                List<org.opencv.core.Point> srcPoints = new LinkedList<>();

                Converters.Mat_to_vector_Point2f(cornersMat.t(), srcPoints);
                srcPoints.remove(3);

                MatOfPoint2f src = new MatOfPoint2f();
                src.fromList(srcPoints);

                Log.i("TEST", String.format("Converted: %s", srcPoints.toString()));
                Log.i("TEST", String.format("Corners: %s\nMat: %s", cornersMat.size().toString(), cornersMat.dump()));
                Log.i("TEST", String.format("Sizes: %s, %s", src.dump(), dest.dump()));

                Mat undistortedImage = new Mat();
                Mat affineTransform = Imgproc.getAffineTransform(src, dest);
                Imgproc.warpAffine(image, undistortedImage, affineTransform, new Size(290, 170));
                // Crop to remove 60 pixels at the end on x axis
                undistortedImage = undistortedImage.submat(0, 170, 0, 220);

//                api.saveMatImage(undistortedImage, "image_" + count + "c_cropped.png");
                Log.i("DETECT", "image_" + count + "c_cropped.png");
                areaImages.add(undistortedImage);
                arucoCorners.add(cornersMat);

                List<String> detection = processItems(objectDetection(undistortedImage));
                if (detection.size() > 0) {
                    if (count != 5) {
                        foundItems.add(detection.get(0));
                        api.setAreaInfo(count, detection.get(0), detection.size());
                    } else {
                        targetItem = detection.get(0);
                        Log.i("TEST", "Target is: " + targetItem);

                        // Let's notify the astronaut when you recognize it.
                        api.notifyRecognitionItem();
                    }
                }

//                Mat transformVec = new Mat();
//                Mat rvec = new Mat();
//                getArucoOffset(arucoCorners.get(count - 1), transformVec, rvec);
//
//                double[] xyzD = transformVec.get(0, 0);
//                float[] xyz = { (float)xyzD[0], (float)xyzD[1], (float)xyzD[2] };
//
//                Kinematics currKin = api.getRobotKinematics();
//                Point offset = new Point(xyz[0], xyz[2], xyz[1]);
//                Log.i("TEST", "Adjusting Offset from Item");
//
//                transformVec.convertTo(transformVec, CvType.CV_64F);
//
//                Mat tvecWorld = transformTvecToWorldFrame(transformVec, currKin.getOrientation());
//                Mat rotationMatrix = rvecToRotationMatrix(rvec);
//                Point target = new Point(tvecWorld.get(0, 0)[0], tvecWorld.get(1, 0)[0], tvecWorld.get(2, 0)[0]);
//                Quaternion orient = rotationMatrixToQuaternion(rotationMatrix);
//
//                Log.i("TEST", "New Tvec: " + target.toString());
//
//                api.moveTo(target, currKin.getOrientation(), true);
//
//                for (int i = 0; i < 3; i++) {
//                    Log.i("TEST", "    " + i + ": " + xyz[i]);
//                }

//                return processItems(objectDetection(undistortedImage));
            } else {
                throw new NoArucoException("No Aruco Tags found. Retake image.");
            }
        } else {
            Log.i("TEST", "IMAGE is NULL!?");
        }

//        return null;
    }

    private String quatToString(Quat4f quaternion) {
        return "X: "+ quaternion.getX() +
                ", Y: " + quaternion.getY() +
                ", Z: " + quaternion.getZ() +
                ", W: "+ quaternion.getW();
    }

    private String quatToString(Quaternion quaternion) {
        return "X: "+ quaternion.getX() +
                ", Y: " + quaternion.getY() +
                ", Z: " + quaternion.getZ() +
                ", W: "+ quaternion.getW();
    }

    private void faceAruco(List<Mat> corners) {
        Quaternion faceAruco = getArucoFacing(corners);
        api.moveTo(api.getRobotKinematics().getPosition(), faceAruco, false);
    }

    private Quaternion getArucoFacing(List<Mat> corners) {
        Mat rvecs = new Mat();
        Mat tvecs = new Mat();

        Aruco.estimatePoseSingleMarkers(corners, 0.05f, camMatrix, distCoeffs, rvecs, tvecs);

        Vector3f vmTvec = new Vector3f((float)tvecs.get(0, 0)[0], (float)tvecs.get(0, 0)[1], (float)tvecs.get(0, 0)[2]);

        Vector3f camFwd = new Vector3f(0f, 0f ,1f);

        Log.i("TEST", "TVec: " + vmTvec.toString());
        vmTvec.normalize();
        Log.i("TEST", "NormTVec: " + vmTvec.toString());

        Vector3f rotAxis = new Vector3f();
        rotAxis.cross(camFwd, vmTvec);
        rotAxis.normalize();

        float rotAngle = (float) Math.acos(camFwd.dot(vmTvec));

        AxisAngle4f orientRot = new AxisAngle4f(rotAxis.z, rotAxis.x, rotAxis.y, rotAngle);

        // START: Chain of Transforms

        Quaternion currentOrientation = api.getRobotKinematics().getOrientation();
        Quat4f astrobeeToOrientation = new Quat4f(currentOrientation.getX(), currentOrientation.getY(), currentOrientation.getZ(), currentOrientation.getW());

        Quat4f cameraToTvec = new Quat4f();
        cameraToTvec.set(orientRot);

        // END

        Log.i("TEST", "CurrentOrient: " + quatToString(api.getRobotKinematics().getOrientation()));

        astrobeeToOrientation.mul(cameraToTvec);
        Log.i("TEST", "cameraToTvec: " + quatToString(astrobeeToOrientation));

        Quaternion faceAruco = new Quaternion(astrobeeToOrientation.x, astrobeeToOrientation.y, astrobeeToOrientation.z, astrobeeToOrientation.w);
        Log.i("TEST", "To Quat: " + quatToString(faceAruco));

        return faceAruco;
    }

    private List<String> processItems(List<Detection> objects) {
        List<String> items = new LinkedList<>();
        for (Detection object : objects) {
            String label = object.getCategories().get(0).getLabel();
            Float score = object.getCategories().get(0).getScore();
            items.add(label);
        }

        // All items should be of same label. So if a false positive is detected, remove
        // it
        // If more than 2 items are detected, make sure all of them are of the same
        // label
        // if one of them differ from majority label, remove it
        if (items.size() > 2) {
            Map<String, Integer> counts = new HashMap<>();
            for (String item : items) {
                counts.put(item, counts.getOrDefault(item, 0) + 1);
            }

            // Remove item if it's count is 1
            for (String key : counts.keySet()) {
                if (counts.get(key) == 1) {
                    items.remove(key);
                }
            }

            // Apply most common label to all items
            Map.Entry<String, Integer> maxEntry = null;
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                if (maxEntry == null || entry.getValue().compareTo(maxEntry.getValue()) > 0) {
                    maxEntry = entry;
                }
            }
            String majority = maxEntry.getKey();

            for (int j = 0; j < items.size(); j++) {
                if (!items.get(j).equals(majority)) {
                    items.set(j, majority);
                }
            }
        }
        return items;
    }

    private List<Detection> objectDetection(Mat image) {
        Mat temp = new Mat(image.size(), CvType.CV_8UC4);
        Imgproc.cvtColor(image, temp, Imgproc.COLOR_BGR2RGBA);
        Bitmap imageBitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(temp, imageBitmap);
        List<Detection> objects = detector.detect(TensorImage.fromBitmap(imageBitmap));
        Log.i("TEST", "Detections:-");
        for (Detection object : objects) {
            Log.i("TEST", "    Detected: " + object.getCategories().get(0).getLabel() + "; Confidence: "
                    + object.getCategories().get(0).getScore());
        }
        return objects;
    }

    private void goToArea(Integer areaIdx) {
        goToArea(areaIdx, null);
    }
    private void goToArea(Integer areaIdx, Quaternion face) {
        /*
         * Figuring Out How to Determine Orientation
         * KIZ 1 - Main Play Area: 10.3 -10.2 4.32 : 11.55 -6.0 5.57
         * if x1 == x2 : Plane/Area is on YZ plane. Side Walls
         * if y1 == y2 : Plane/Area is on XZ plane. Front or Back
         * if z1 == z2 : Plane/Area is on XY plane. Top or Bottom
         * Which wall it is can be determined from proximity of fixed xyz component
         * proximity to KIZ limits
         */
        double[][] kiz1 = {
                { 10.3, 11.55 },
                { -10.2, -6.0 },
                { 4.32, 5.57 }
        };

        Map<String, Quaternion> orientations = new HashMap<>();
        orientations.put("top", new Quaternion());

        double[][][] areas = {
                {
                        { 10.42, 11.48 },
                        { -10.58, -10.58 },
                        { 4.82, 5.57 }
                },
                {
                        { 10.3, 11.55 },
                        { -9.25, -8.5 },
                        { 3.76203, 3.76203 }
                },
                {
                        { 10.3, 11.55 },
                        { -8.4, -7.45 },
                        { 3.76093, 3.76093 }
                },
                {
                        { 9.866984, 9.866984 },
                        { -7.34, -6.365 },
                        { 4.32, 5.57 }
                }
        };

        Point a1 = new Point(10.95d, -10.0d, 5.195d);
        Point a2a = new Point(10.52d, -9.625d, 4.75d);
        Point a2b = new Point(10.925d, -8.875d, 4.45d);
        Point a3 = new Point(10.925d, -7.91d, 4.45d);
        Point a4a = new Point(10.575d, -7.3d, 4.45d);
        Point a4 = new Point(10.5d, -6.8525d, 4.945d);
        Point astronautPos = new Point(11.143d, -6.7607d, 4.9654d);

        Quaternion q1 = new Quaternion(0f, 0f, -0.707f, 0.707f); // Area 1
        Quaternion q2a = new Quaternion(0.271f, 0.271f, -0.653f, 0.653f); // Area 2 Intermediate
        Quaternion q23 = new Quaternion(0.5f, 0.5f, -0.5f, 0.5f); // Area 2 and 3
        Quaternion q4a = new Quaternion(0.707f, 0f, -0.707f, 0f); // Area 4 Intermediate
        Quaternion q4 = new Quaternion(0f, 0f, -1f, 0f); // Area 4
        Quaternion astronautQuat = new Quaternion(0f, 0f, -0.707f, -0.707f); // Facing Astronaut

        // Return path for each area

        Point ra123a = new Point(11.15d, -7.5d, 5.3d);
        Quaternion rq123a = new Quaternion(0.0f, 0.0f, -0.966f, -0.259f);

        Point ra1b = new Point(11.05d, -8.45d, 4.7d);
        Quaternion rq1b = new Quaternion(0f, 0f, -1f, 0f);

        Point ra1c = new Point(11.2d, -9.5d, 5.3d);
        Quaternion rq1c = new Quaternion(0f, 0f, -0.906f, 0.423f);

        List<Point> locations = new ArrayList<>(0);
        List<Quaternion> orients = new ArrayList<>(0);

        switch (areaIdx) {
            case 1:
                locations.add(a1); orients.add(q1);
                break;
            case 2:
                // Skip intermediate point when skip Area 1
                if (!SKIPAREA1) {
                    locations.add(a2a); orients.add(q2a);
                }
                locations.add(a2b); orients.add(q23);
                break;
            case 3:
                locations.add(a3); orients.add(q23);
                break;
            case 4:
                locations.add(a4a); orients.add(q4a);
                locations.add(a4); orients.add(q4);
                break;
            case 5:
                locations.add(astronautPos); orients.add(astronautQuat);
                break;
            // Returning cases
            case 14:
                locations.add(a4); orients.add(q4);
                break;
            case 13:
                locations.add(ra123a); orients.add(rq123a);
                locations.add(a3); orients.add(q23);
                break;
            case 12:
                locations.add(ra123a); orients.add(rq123a);
                locations.add(a2b); orients.add(q23);
                break;
            case 11:
                locations.add(ra123a); orients.add(rq123a);
                locations.add(ra1b); orients.add(rq1b);
                locations.add(ra1c); orients.add(rq1c);
                locations.add(a1); orients.add(q1);
                break;
        }
        if (face != null) {
            orients.set(orients.size() - 1, face);
        }
        for (int i = 0; i < locations.size(); i++) { // Traverse through checkpoints
            api.moveTo(locations.get(i), orients.get(i), false);
        }
    }

    private Float calcDistance(Point a, Point b) {
        return (float) Math.sqrt(
                Math.pow(a.getX() - b.getX(), 2)
                        + Math.pow(a.getY() - b.getY(), 2)
                        + Math.pow(a.getZ() - b.getZ(), 2));
    }

    @Override
    protected void runPlan1() {
        // Create ObjectDetector from transfer trained model
        initDetector();

        // Converting Camera Intrinsics to usable formats
        double[][] intrinsics = api.getNavCamIntrinsics();
        Mat image = new Mat();

        camMatrix = new MatOfDouble();
        camMatrix.fromArray(intrinsics[0]);
        camMatrix.create(new Size(3, 3), CvType.CV_64F);

        distCoeffs = new MatOfDouble();
        distCoeffs.fromArray(intrinsics[1]);

        Log.i("TEST", String.format("Matrix: %s\nCoeffs: %s", camMatrix.dump(), distCoeffs.dump()));

        int count = 0; // To keep track of image name

        // The mission starts.
        Log.i("TEST", "Starting");
        api.startMission();

        /* **************************************************** */
        /* Let's move to the each area and recognize the items. */
        /* **************************************************** */

        Point astronautPos = new Point(11.143d, -6.7607d, 4.9654d);
        Kinematics kin = api.getRobotKinematics();
        Log.i("TEST", "Distance from Astronaut: " + calcDistance(kin.getPosition(), astronautPos));
        if (calcDistance(kin.getPosition(), astronautPos) < 0.1f) {
            Log.i("TEST", "Resetting position to Area 1");
            goToArea(11);
        }


        // Go To Points and Take Pictures

        int startIdx = 1;
        if (SKIPAREA1) {
            areaImages.add(new Mat()); // Blank image for Area 1
            arucoCorners.add(new Mat()); // Blank corners for Area 1
            foundItems.add("");
            startIdx = 2;
        }

        for (int i = startIdx; i <= 5; i++) {
            // Move to ith Area
            goToArea(i);
            Log.i("TEST", String.format("FACING Area %d", i));
            if (i == 5) {
                // When you move to the front of the astronaut, report the rounding completion.
                api.reportRoundingCompletion();
            }

            // Get a camera image.
            image = api.getMatNavCam();

            // Get facing quaternion before undistorting image
            List<Mat> corners = new ArrayList<>();
            Mat ids = new MatOfInt();
            Aruco.detectMarkers(image, Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250), corners, ids);
            if (corners.size() > 0) {
                faceOrients.add(getArucoFacing(corners));
            } else {
                faceOrients.add(null);
            }

            image = undistort(image, i);

            try {
                detectAndSaveAruco(image, i);
            } catch (NoArucoException e) {
                Log.i("TEST", "Failing Aruco at Area " + i + ". Trying again.");
                image = api.getMatNavCam();
                image = undistort(image, i);
                try {
                    detectAndSaveAruco(image, i);
                } catch (NoArucoException e1) {
                    Log.i("TEST", "Failing Aruco at Area " + i + ". Trying again.");
                    image = api.getMatNavCam();
                    image = undistort(image, i);
                    try {
                        detectAndSaveAruco(image, i);
                    } catch (NoArucoException e2) {
                        Log.i("TEST", "Failing Aruco at Area " + i + ". Trying again.");
                        image = api.getMatNavCam();
                        image = undistort(image, i);
                        try {
                            detectAndSaveAruco(image, i);
                        } catch (NoArucoException e3) {
                            Log.i("TEST", "Failing Aruco at Area " + i + ". Trying again.");
                            image = api.getMatNavCam();
                            image = undistort(image, i);
                            try {
                                detectAndSaveAruco(image, i);
                            } catch (NoArucoException e4) {
                                Log.i("TEST", "Failing Aruco at Area " + i + ". Trying again.");
                                image = api.getMatNavCam();
                                image = undistort(image, i);
                                try {
                                    detectAndSaveAruco(image, i);
                                } catch (NoArucoException e5) {
                                    Log.i("TEST", "No Aruco at Area " + i);
                                }
                            }
                        }
                    }
                }
            }
        }


        /* ********************************************************** */
        /* Write your code to recognize which item the astronaut has. */
        /* ********************************************************** */

        for (int k = 0; k < foundItems.size(); k++) {
            if (targetItem.equalsIgnoreCase(foundItems.get(k))) { // Return to Area 1 takes too long, so ignore it.
                Log.i("TEST", "GOING TO TARGET");

                try { // Go to Target Area with orientation towards Cam
                    goToArea(10 + k + 1, faceOrients.get(k));
                } catch (Exception ex) { // If orientation fails, just go to area if area not 1.
                    if (k != 0) {
                        goToArea(10 + k + 1);
                    }
                }
            }
        }


        /*
         * *****************************************************************************
         * **************************
         */
        /*
         * Write your code to move Astrobee to the location of the target item (what the
         * astronaut is looking for)
         */
        /*
         * *****************************************************************************
         * **************************
         */

        // Move closer to image using Aruco and relative move

        // Take a snapshot of the target item.
        api.takeTargetItemSnapshot();
    }

    @Override
    protected void runPlan2() {
        double[][] intrinsics = api.getNavCamIntrinsics();

        camMatrix = new MatOfDouble();
        camMatrix.fromArray(intrinsics[0]);
        camMatrix.create(new Size(3, 3), CvType.CV_64F);

        distCoeffs = new MatOfDouble();
        distCoeffs.fromArray(intrinsics[1]);

        api.startMission();

        // Reset Look

        AxisAngle4f vmEuler = new AxisAngle4f(1f, 0f, 0f, 0f);

        Quat4f vmEulerQuat = new Quat4f();
        vmEulerQuat.set(vmEuler);

        Quaternion targetQuat = new Quaternion(0f, 0f, 0f, 1f);
//        Quaternion targetQuat = new Quaternion(vmEulerQuat.getX(), vmEulerQuat.getY(), vmEulerQuat.getZ(), vmEulerQuat.getW());

//        Log.i("TEST", "Reset Look");
//        api.moveTo(api.getRobotKinematics().getPosition(), targetQuat, false);

        // Go To Area
        int targetArea = 5;

        Log.i("TEST", "Go to Area " + targetArea);

        goToArea(targetArea);

        Mat image = api.getMatNavCam();
        List<Mat> corners = new ArrayList<>();
        Mat ids = new MatOfInt();
        Mat detectionImage = undistort(image, 1);
        Aruco.detectMarkers(image, Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250), corners, ids);
        Log.i("TEST", "Detected Markers: " + corners.size());

        Quat4f targetOrientation = new Quat4f();
        faceAruco(corners);
//        getArucoOffset(corners.get(0), detectionImage, targetOrientation);

//        Quaternion currOrientation = api.getRobotKinematics().getOrientation();
//        Quat4f vmCurrOrientation = new Quat4f(currOrientation.getX(), currOrientation.getY(), currOrientation.getZ(), currOrientation.getW());
//        Log.i("TEST", "Current Orientation: " + quatToString(currOrientation));
//        Log.i("TEST", "Current VmOrientation: " + quatToString(vmCurrOrientation));
//        vmCurrOrientation.mul(targetOrientation);
//        Log.i("TEST", "Target Orientation: " + quatToString(vmCurrOrientation));
//
//        targetQuat = new Quaternion(vmCurrOrientation.getX(), vmCurrOrientation.getY(), vmCurrOrientation.getZ(), vmCurrOrientation.getW());
//
//        api.moveTo(api.getRobotKinematics().getPosition(), targetQuat, false);

        Log.i("TEST", "Facing Aruco");
    }

    @Override
    protected void runPlan3() {
        // write your plan 3 here.
    }

    // You can add your method.
    private String yourMethod() {
        return "your method";
    }

    // Convert rvec to rotation matrix
    public Mat rvecToRotationMatrix(Mat rvec) {
        Mat rotationMatrix = new Mat(3, 3, CvType.CV_64F);
        Calib3d.Rodrigues(rvec, rotationMatrix);
        return rotationMatrix;
    }

    // Convert rotation matrix to quaternion
    public Quaternion rotationMatrixToQuaternion(Mat rotationMatrix) {
        double[] r = new double[9];
        rotationMatrix.get(0, 0, r);

        double m00 = r[0], m01 = r[1], m02 = r[2];
        double m10 = r[3], m11 = r[4], m12 = r[5];
        double m20 = r[6], m21 = r[7], m22 = r[8];

        float qw, qx, qy, qz;

        float trace = (float) (m00 + m11 + m22);
        if (trace > 0) {
            float s = 0.5f / (float) Math.sqrt(trace + 1.0);
            qw = 0.25f / s;
            qx = (float) ((m21 - m12) * s);
            qy = (float) ((m02 - m20) * s);
            qz = (float) ((m10 - m01) * s);
        } else {
            if (m00 > m11 && m00 > m22) {
                float s = 2.0f * (float) Math.sqrt(1.0 + m00 - m11 - m22);
                qw = (float) ((m21 - m12) / s);
                qx = 0.25f * s;
                qy = (float) ((m01 + m10) / s);
                qz = (float) ((m02 + m20) / s);
            } else if (m11 > m22) {
                float s = 2.0f * (float) Math.sqrt(1.0 + m11 - m00 - m22);
                qw = (float) ((m02 - m20) / s);
                qx = (float) ((m01 + m10) / s);
                qy = 0.25f * s;
                qz = (float) ((m12 + m21) / s);
            } else {
                float s = 2.0f * (float) Math.sqrt(1.0 + m22 - m00 - m11);
                qw = (float) ((m10 - m01) / s);
                qx = (float) ((m02 + m20) / s);
                qy = (float) ((m12 + m21) / s);
                qz = 0.25f * s;
            }
        }

        return new Quaternion(qx, qy, qz, qw);
    }

    // Assuming Q is the quaternion representing the robot's current orientation
    public Mat quaternionToRotationMatrix(Quaternion quaternion) {
        float qx = quaternion.getX();
        float qy = quaternion.getY();
        float qz = quaternion.getZ();
        float qw = quaternion.getW();

        Mat rotationMatrix = Mat.eye(3, 1, CvType.CV_64F);
        rotationMatrix.put(0, 0,
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw);
        rotationMatrix.put(1, 0,
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw);
        rotationMatrix.put(2, 0,
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy);
        return rotationMatrix;
    }

    public Mat transformTvecToWorldFrame(Mat tvec, Quaternion quaternion) {
        Mat rotationMatrix = quaternionToRotationMatrix(quaternion);
        Mat tvecWorld = new Mat();
        Mat tqvec = new Mat(1, 3, CvType.CV_64F);
        tqvec.put(0, 0, tvec.get(0, 0)[0]);
        tqvec.put(0, 1, tvec.get(0, 0)[1]);
        tqvec.put(0, 2, tvec.get(0, 0)[2]);
        Log.i("TEST", "RV: " + rotationMatrix.toString() + " TV: " + tqvec.toString());
        Core.gemm(rotationMatrix, tqvec, 1d, new Mat(), 0d, tvecWorld);
        return tvecWorld;
    }

}
