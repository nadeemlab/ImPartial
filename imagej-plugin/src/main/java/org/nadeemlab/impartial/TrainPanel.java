package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;
import java.awt.*;

public class TrainPanel extends JPanel {
    private final JTextField epochsField;
    private final JTextField patchesField;
    private final JTextField patienceField;
    private final JButton startStopButton;

    TrainPanel(ImpartialController controller) {

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);
        setBorder(BorderFactory.createEmptyBorder(0, 5, 5, 5));

        JLabel title = new JLabel("Train");
        title.setFont(new Font("sans-serif", Font.PLAIN, 15));
        add(title);
        add(Box.createVerticalStrut(10));

        JPanel configPanel = new JPanel(new GridBagLayout());
        configPanel.setAlignmentX(LEFT_ALIGNMENT);

        epochsField = new JTextField("10", 3);
        epochsField.setToolTipText("Max number of epochs to train");

        patchesField = new JTextField("200", 3);
        patchesField.setToolTipText("Number of patches sampled per epoch");

        patienceField = new JTextField("10", 3);
        patienceField.setToolTipText("Number of times the evaluation loss can no-decrease before the training stops");

        addRow(configPanel, "Epochs", epochsField);
        addRow(configPanel, "Patches", patchesField);
        addRow(configPanel, "Patience", patienceField);

        startStopButton = new JButton("Train");
        startStopButton.setEnabled(false);

        startStopButton.addActionListener(e -> {
            if (startStopButton.getText().equals("Train")) {
                controller.startTraining();
                startStopButton.setText("Stop");
            } else {
                controller.stopTraining();
                startStopButton.setText("Train");
            }
        });

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        buttonPanel.setAlignmentX(LEFT_ALIGNMENT);
        buttonPanel.add(startStopButton);

        add(configPanel);
        add(buttonPanel);
    }

    private void addRow(JPanel panel, String labelText, JTextField textField) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = GridBagConstraints.RELATIVE;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 0, 5, 0);

        JLabel label = new JLabel(labelText);
        panel.add(label, c);

        c.gridx = 1;
        c.gridy = GridBagConstraints.RELATIVE;
        c.weightx = 1;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 5, 5, 0);

        panel.add(textField, c);
    }


    public JSONObject getTrainParams() {
        JSONObject params = new JSONObject();

        params.put("max_epochs", Integer.parseInt(epochsField.getText()));
        params.put("npatches_epoch", Integer.parseInt(patchesField.getText()));
        params.put("early_stop_patience", Integer.parseInt(patienceField.getText()));

        return params;
    }

    public void onStarted() {
        startStopButton.setEnabled(true);
    }

    public void onStopped() {
        startStopButton.setEnabled(false);
    }

    public void onTrainingStopped() {
        startStopButton.setText("Train");
    }
}
