package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;
import java.awt.*;

public class TrainPanel extends JPanel {
    private final JPanel epochsPanel;
    private final JPanel patchesPanel;
    private final JPanel patiencePanel;
    private final JButton startStopButton;

    TrainPanel(ImpartialController controller) {

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);

        setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("TRAIN"),
            BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel configPanel = new JPanel();
        configPanel.setLayout(new BoxLayout(configPanel, BoxLayout.Y_AXIS));
        configPanel.setAlignmentX(LEFT_ALIGNMENT);

        epochsPanel = createParamPanel("epochs", 10);
        epochsPanel.setToolTipText("max number of epochs to train");

        patchesPanel = createParamPanel("patches", 200);
        patchesPanel.setToolTipText("number of patches sampled per epoch");

        patiencePanel = createParamPanel("patience", 10);
        patiencePanel.setToolTipText("number of times the evaluation loss can no-decrease before the training stops");

        configPanel.add(epochsPanel);
        configPanel.add(patchesPanel);
        configPanel.add(patiencePanel);

        startStopButton = new JButton("train");
        startStopButton.setEnabled(false);

        startStopButton.addActionListener(e -> {
            if (startStopButton.getText().equals("train")) {
                controller.startTraining();
                startStopButton.setText("stop");
            } else {
                controller.stopTraining();
                startStopButton.setText("train");
            }
        });

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.setAlignmentX(LEFT_ALIGNMENT);
        buttonPanel.add(startStopButton);

        add(configPanel);
        add(buttonPanel);
    }

    private JPanel createParamPanel(String name, int value) {
        JPanel paramPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        paramPanel.setAlignmentX(LEFT_ALIGNMENT);
        paramPanel.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));

        paramPanel.add(new JLabel(String.format("<html> <strong>%s</strong> &nbsp;", name)));
        paramPanel.add(new JTextField(String.valueOf(value), 3));

        return paramPanel;
    }

    public JSONObject getTrainParams() {
        JSONObject params = new JSONObject();

        JTextField epochs = (JTextField) epochsPanel.getAccessibleContext().getAccessibleChild(1);
        JTextField patches = (JTextField) patchesPanel.getAccessibleContext().getAccessibleChild(1);
        JTextField patience = (JTextField) patiencePanel.getAccessibleContext().getAccessibleChild(1);

        params.put("max_epochs", Integer.parseInt(epochs.getText()));
        params.put("npatches_epoch", Integer.parseInt(patches.getText()));
        params.put("early_stop_patience", Integer.parseInt(patience.getText()));

        return params;
    }

    public void onStarted() {
        startStopButton.setEnabled(true);
    }

    public void onStopped() {
        startStopButton.setEnabled(false);
    }

    public void onTrainingStopped() {
        startStopButton.setText("train");
    }
}
