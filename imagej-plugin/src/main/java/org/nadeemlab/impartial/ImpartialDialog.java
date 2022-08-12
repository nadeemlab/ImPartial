package org.nadeemlab.impartial;

import org.json.JSONObject;
import ij.plugin.frame.RoiManager;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.display.ImageDisplayService;
import net.imagej.display.OverlayService;
import net.imagej.ops.OpService;
import net.imagej.roi.ROIService;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.display.DisplayService;
import org.scijava.io.IOService;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;

import javax.swing.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ImpartialDialog {

    @Parameter
    private OpService ops;
    @Parameter
    private LogService log;
    @Parameter
    private StatusService status;
    @Parameter
    private CommandService cmd;
    @Parameter
    private ThreadService thread;
    @Parameter
    private DatasetIOService datasetIOService;
    @Parameter
    private ImageDisplayService imageDisplayService;
    @Parameter
    private DisplayService displayService;
    @Parameter
    private UIService ui;
    @Parameter
    private IOService io;
    @Parameter
    private ROIService roiService;
    @Parameter
    private OverlayService overlayService;

    private final MonaiLabelClient monaiClient = new MonaiLabelClient();
    private final DatasetPanel datasetPanel;
    JPanel mainPane;

    protected JLabel actionLabel;
    private final File labelFile;
    private final File imageFile;
    private final File outputFile;
    private String imageId;

    /**
     * Create the dialog.
     */
    public ImpartialDialog(final Context context) {
        context.inject(this);

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

        mainPane = new JPanel();
        mainPane.setLayout(new BoxLayout(mainPane, BoxLayout.PAGE_AXIS));

        ServerPanel serverPanel = new ServerPanel(this, monaiClient);
        mainPane.add(serverPanel);

        datasetPanel = new DatasetPanel(this, monaiClient);
        mainPane.add(datasetPanel);

        ModelPanel modelPanel = new ModelPanel(this);
        mainPane.add(modelPanel);

    }

    public void setImageId(String imageId) {
        this.imageId = imageId;
    }

    public void connect() {
        datasetPanel.populateSampleList();
    }

    public void loadLabel() {
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
        RoiManager.getRoiManager().runCommand("Save", labelFile.getAbsolutePath());

        monaiClient.putDatastoreLabel(imageId, labelFile.getAbsolutePath());
    }

    public void showImage() {
        byte[] imageBytes = monaiClient.getDatastoreImage(imageId);

        try {
            FileOutputStream stream = new FileOutputStream(imageFile.getAbsolutePath());
            stream.write(imageBytes);
            stream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Dataset image = null;
        try {
            image = datasetIOService.open(imageFile.getAbsolutePath());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

//		TODO: is there a way of using the same display instead
//		of closing it and opening a new one
        if (displayService.getActiveDisplay() != null) {
            displayService.getActiveDisplay().close();
        }

        ui.show(image);
    }

    public void infer() {
        byte[] output = monaiClient.postInfer("impartial", imageId);

        try {
            FileOutputStream stream = new FileOutputStream(outputFile);
            stream.write(output);
            stream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        RoiManager.getRoiManager().runCommand("Open", outputFile.getAbsolutePath());
    }

    public void train() {
        monaiClient.deleteTrain();
        monaiClient.postTrain("impartial");

        TrainProgress trainProgress = new TrainProgress(status, monaiClient);
        trainProgress.monitorTraining();
    }

    public void showNextSample() {
        JSONObject res = monaiClient.postActiveLearning("random");
        imageId = res.getString("id");
        showImage();
    }

}
