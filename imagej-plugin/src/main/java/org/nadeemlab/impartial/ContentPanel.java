package org.nadeemlab.impartial;

import ij.plugin.frame.RoiManager;
import io.scif.services.DatasetIOService;
import net.imagej.Dataset;
import org.json.JSONObject;
import org.scijava.app.StatusService;
import org.scijava.display.DisplayService;
import org.scijava.ui.UIService;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class ContentPanel extends JPanel {
    private MonaiLabelClient monaiClient;
    private DisplayService displayService;
    private StatusService status;
    private DatasetIOService datasetIOService;
    private UIService ui;
    protected JLabel actionLabel;
    private final File labelFile;
    private final File imageFile;
    private final File outputFile;
    private String imageId;

    ContentPanel(MonaiLabelClient monaiClient, DisplayService displayService, StatusService status,
                 DatasetIOService datasetIOService, UIService ui) {
        this.monaiClient = monaiClient;

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

        setLayout(new FlowLayout());
        setBorder(new EmptyBorder(5, 5, 5, 5));

        {
            final JButton btnNextSampleButton = new JButton("Next sample");
            btnNextSampleButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    showNextSample();
                }
            });
            add(btnNextSampleButton);
        }
        {
//            String[] samples = getDatastoreSamples();
//            Arrays.sort(samples);

//            final JComboBox selectSample = new JComboBox(samples);
            final JComboBox selectSample = new JComboBox();
            selectSample.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent e) {
                    JComboBox cb = (JComboBox) e.getSource();
                    imageId = (String) cb.getSelectedItem();
                    showImage(imageId);
                }
            });
            add(selectSample);
        }
        {
            final JButton btnInferButton = new JButton("Infer");
            btnInferButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    infer();
                }
            });
            add(btnInferButton);
        }
        {
            final JButton btnSubmitLabelButton = new JButton("Submit label");
            btnSubmitLabelButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    submitLabel();
                }
            });
            add(btnSubmitLabelButton);
        }
        {
            final JButton btnLoadLabelButton = new JButton("Load label");
            btnLoadLabelButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    loadLabel();
                }
            });
            add(btnLoadLabelButton);
        }
        {
            final JButton btnTrainButton = new JButton("Train");
            btnTrainButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(final ActionEvent arg0) {
                    train();
                }
            });
            add(btnTrainButton);
        }
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
}
