package org.nadeemlab.impartial;

import ij.ImagePlus;
import ij.io.Opener;
import ij.plugin.frame.RoiManager;
import ij.process.FloatProcessor;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.binary.Thresholder;
import net.imglib2.algorithm.labeling.ConnectedComponents;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import org.json.JSONArray;
import org.json.JSONObject;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.plugin.Parameter;
import org.scijava.ui.UIService;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Hashtable;
import java.util.stream.StreamSupport;

public class ImpartialController {
    private final JFrame mainFrame;
    private final ImpartialContentPane contentPane;
    private final JDialog imageDialog;
    private final JLabel imageLabel = new JLabel();
    private File imageFile;
    private File labelFile;
    private File outputFile;
    private final MonaiLabelClient monaiClient = new MonaiLabelClient();
    private final Hashtable<String, JSONObject> modelOutputs = new Hashtable<>();
    private int currentEpoch = 0;
    @Parameter
    private DatasetIOService datasetIOService;
    @Parameter
    private UIService ui;
    @Parameter
    private StatusService status;
    @Parameter
    private OpService ops;

    public ImpartialController(final Context context) {
        context.inject(this);
        createTempFiles();

        mainFrame = new JFrame("ImPartial");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Create and set up the content pane.
        contentPane = new ImpartialContentPane(this);
        contentPane.setOpaque(true); //content panes must be opaque
        mainFrame.setContentPane(contentPane);

        imageDialog = new JDialog(mainFrame, false);
        imageDialog.add(imageLabel);

        mainFrame.pack();
        mainFrame.setVisible(true);
        mainFrame.setVisible(true);
    }

    private void createTempFiles() {
        try {
            imageFile = File.createTempFile("impartial-image-", ".png");
            imageFile.deleteOnExit();

            labelFile = File.createTempFile("impartial-label-", ".zip");
            labelFile.deleteOnExit();

            outputFile = File.createTempFile("impartial-output-", ".zip");
            outputFile.deleteOnExit();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void connect() {
        String[] samples = getDatastoreSamples();
        contentPane.populateSampleList(samples);
    }

    public void updateImage(String imageId) {
        displayImage(imageId);

        contentPane.enableInferButton();
        if (modelOutputs.containsKey(imageId)) {
            JSONObject output = modelOutputs.get(imageId);
            contentPane.updateInferInfo(
                    output.getInt("epoch"),
                    output.getString("time")
            );
            contentPane.updateInferView(true);
        }
        else {
            contentPane.disableInferInfo();
            contentPane.updateInferView(false);
        }
    }

    public void displayImage(String imageId) {
        retrieveImage(imageId);
        displayStoredImage();
    }

    private void retrieveImage(String imageId) {
        byte[] imageBytes = monaiClient.getDatastoreImage(imageId);

        try {
            FileOutputStream stream = new FileOutputStream(imageFile.getAbsolutePath());
            stream.write(imageBytes);
            stream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void displayStoredImage() {
        final ImagePlus imp = new Opener().openImage(imageFile.getAbsolutePath());
        setDialogImage(imp.getBufferedImage());
    }

    public void setDialogImage(BufferedImage img) {
        imageLabel.setIcon(new ImageIcon(img));
        imageDialog.setTitle(contentPane.getSelectedImageId());
        imageDialog.pack();

        if (!imageDialog.isVisible()) {
            Point mainFrameLocation = mainFrame.getLocation();
            imageDialog.setLocation(
                    mainFrameLocation.x + mainFrame.getSize().width,
                    mainFrameLocation.y
            );
            imageDialog.setVisible(true);
        }
    }

    public void displayStoredDataset() {
        Dataset image;
        try {
            image = datasetIOService.open(imageFile.getAbsolutePath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        ui.show(image);
    }

    public void setMonaiClientUrl(URL url) {
        monaiClient.setUrl(url);
    }

    public String[] getDatastoreSamples() {
        JSONObject datastore = monaiClient.getDatastore();
        Iterable<String> iterable = () -> datastore.getJSONObject("objects").keys();

        return StreamSupport.stream(iterable.spliterator(), false)
                .toArray(String[]::new);
    }

    public void loadLabel() {
        displayStoredDataset();

        String imageId = contentPane.getSelectedImageId();
        byte[] label = monaiClient.getDatastoreLabel(imageId);

        try {
            FileOutputStream stream = new FileOutputStream(labelFile);
            stream.write(label);
            stream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        RoiManager.getRoiManager().runCommand("Open", labelFile.getAbsolutePath());
    }

    public void submitLabel() {
        String imageId = contentPane.getSelectedImageId();
        RoiManager.getRoiManager().runCommand("Save", labelFile.getAbsolutePath());
        monaiClient.putDatastoreLabel(imageId, labelFile.getAbsolutePath());
    }

    public JSONObject getImageInfo(String imageId) {
        JSONObject datastore = monaiClient.getDatastore();
        return datastore.getJSONObject("objects").getJSONObject(imageId);
    }

    public void train() {
        monaiClient.deleteTrain();

        JSONObject params = contentPane.getTrainParams();

        monaiClient.postTrain("impartial", params);

        TrainProgress.monitorTraining(this);
    }

    public void infer() {
        String imageId = contentPane.getSelectedImageId();
        JSONObject params = new JSONObject();
        params.put("threshold", (Float) contentPane.getThreshold());

        JSONObject modelOutput = monaiClient.postInferJson("impartial", imageId, params);

        modelOutput.put("epoch", currentEpoch);

        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm");
        LocalDateTime time = LocalDateTime.now();
        modelOutput.put("time", time.format(formatter));

        modelOutputs.put(imageId, modelOutput);

        contentPane.updateInferInfo(
                modelOutput.getInt("epoch"),
                modelOutput.getString("time")
        );
        contentPane.updateInferView(true);
    }

    public void displayInfer() {
        String imageId = contentPane.getSelectedImageId();
        ImagePlus imp = jsonArrayToImp(modelOutputs.get(imageId).getJSONArray("output"));

        Img<FloatType> img = ImageJFunctions.wrapFloat(imp);

        FloatType threshold = new FloatType(contentPane.getThreshold());
        Img<BitType> binaryImg = Thresholder.threshold(img, threshold, true, 2);

        final long[] dims = new long[binaryImg.numDimensions()];
        binaryImg.dimensions(dims);

        final RandomAccessibleInterval<BitType> outline = ArrayImgs.bits(dims);
        ops.morphology().outline(outline, binaryImg, true);

        final RandomAccessibleInterval<UnsignedShortType> indexImg = ArrayImgs.unsignedShorts(dims);
        final ImgLabeling<Integer, UnsignedShortType> labeling = new ImgLabeling<>(indexImg);
        ops.labeling().cca(labeling, outline, ConnectedComponents.StructuringElement.FOUR_CONNECTED);

        ImagePlus output = ImageJFunctions.wrapFloat(labeling.getIndexImg(), "output");

        imageLabel.setIcon(new ImageIcon(output.getBufferedImage()));
    }

    public void displayEntropy() {
        String imageId = contentPane.getSelectedImageId();
        ImagePlus entropy = jsonArrayToImp(modelOutputs.get(imageId).getJSONArray("entropy"));
        imageLabel.setIcon(new ImageIcon(entropy.getBufferedImage()));
    }

    public ImagePlus jsonArrayToImp(JSONArray input) {
        int width = input.length();
        int height = input.getJSONArray(0).length();
        float[][] output = new float[width][height];

        for (int i = 0; i < input.length(); i++) {
            JSONArray row = input.getJSONArray(i);
            for (int j = 0; j < row.length(); j++) {
                output[j][i] = (float) row.getDouble(j);
            }
        }

        FloatProcessor fp = new FloatProcessor(output);
        return new ImagePlus("output", fp);
    }

    public void showStatus(int progress, int max, String message) {
        status.showStatus(progress, max, message);
    }

    public JSONObject getInfo() {
        return monaiClient.getInfo();
    }

    public JSONObject getTrain() {
        return monaiClient.getTrain();
    }

}
