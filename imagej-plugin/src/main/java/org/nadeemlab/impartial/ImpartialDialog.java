package org.nadeemlab.impartial;


import ij.plugin.frame.RoiManager;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import net.imagej.display.ImageDisplayService;
import net.imagej.display.OverlayService;
import net.imagej.ops.OpService;
import net.imagej.roi.ROIService;
import net.imglib2.type.numeric.RealType;
import org.json.JSONObject;
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
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;


public class ImpartialDialog<T extends RealType<T>> extends JDialog {

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
    private final JPanel contentPanel = new JPanel();
    protected JLabel actionLabel;
    private final File labelFile;
    private final File imageFile;
    private final File outputFile;
    private String imageId;

    /**
     * Create the dialog.
     */
    public ImpartialDialog(final Context ctx) {

        try {
            labelFile = File.createTempFile("impartial-label-", ".zip");
            labelFile.deleteOnExit();

            imageFile = File.createTempFile("impartial-image-", ".png");
            imageFile.deleteOnExit();

            outputFile = File.createTempFile("impartial-output-", ".zip");
            outputFile.deleteOnExit();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        ctx.inject(this);

        thread.setExecutorService(Executors.newScheduledThreadPool(1));

        setBounds(100, 100, 150, 300);
        getContentPane().setLayout(new BorderLayout());
        contentPanel.setLayout(new FlowLayout());
        contentPanel.setBorder(new EmptyBorder(5, 5, 5, 5));
        getContentPane().add(contentPanel, BorderLayout.CENTER);

        //Create a label to put messages during an action event.
        actionLabel = new JLabel("0");
//		actionLabel.setBorder(BorderFactory.createEmptyBorder(10, 0, 0, 0));
        contentPanel.add(actionLabel);

        {
            final JButton btnNextSampleButton = new JButton("Next sample");
            btnNextSampleButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    showNextSample();
                }
            });
            contentPanel.add(btnNextSampleButton);
        }
        {
            JSONObject datastore = monaiClient.getDatastore();

            List<String> samplesList = new ArrayList<>();
            for (Iterator<String> it = datastore.getJSONObject("objects").keys(); it.hasNext(); ) {
                samplesList.add(it.next());
            }
            String[] samples = Arrays.copyOf(samplesList.toArray(), samplesList.size(), String[].class);

            Arrays.sort(samples);

            final JComboBox selectSample = new JComboBox(samples);
            selectSample.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent e) {
                    JComboBox cb = (JComboBox) e.getSource();
                    imageId = (String) cb.getSelectedItem();
                    showImage(imageId);
                }
            });
            contentPanel.add(selectSample);
        }
        {
            final JButton btnInferButton = new JButton("Infer");
            btnInferButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    infer();
                }
            });
            contentPanel.add(btnInferButton);
        }
        {
            final JButton btnSubmitLabelButton = new JButton("Submit label");
            btnSubmitLabelButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    submitLabel();
                }
            });
            contentPanel.add(btnSubmitLabelButton);
        }
        {
            final JButton btnLoadLabelButton = new JButton("Load label");
            btnLoadLabelButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    loadLabel();
                }
            });
            contentPanel.add(btnLoadLabelButton);
        }
        {
            final JButton btnTrainButton = new JButton("Train");
            btnTrainButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    train();
                }
            });
            contentPanel.add(btnTrainButton);
        }
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

        JSONObject res = monaiClient.putDatastoreLabel(imageId, labelFile.getAbsolutePath());
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

        showImage(imageId);
    }

    private void showImage(String imageId) {
        actionLabel.setText(imageId);

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
}
