package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;

public class TrainPanel extends JPanel {
    private ImpartialController controller;
    private final JPanel epochsPanel;
    private final JPanel nchannelsPanel;
    private final JPanel patchesPanel;
    private final JPanel patiencePanel;

    TrainPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("train"),
            BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.PAGE_AXIS));

        JPanel configPanel = new JPanel();
        configPanel.setLayout(new BoxLayout(configPanel, BoxLayout.PAGE_AXIS));

        nchannelsPanel = createParamPanel("nchannels", 1);
        nchannelsPanel.setToolTipText("number of channels for dataset");

        epochsPanel = createParamPanel("epochs", 10);
        epochsPanel.setToolTipText("max number of epochs to train");

        patchesPanel = createParamPanel("patches", 200);
        patchesPanel.setToolTipText("number of patches sampled per epoch");

        patiencePanel = createParamPanel("patience", 10);
        patiencePanel.setToolTipText("number of times the evaluation loss can no-decrease before the training stops");

        configPanel.add(nchannelsPanel);
        configPanel.add(epochsPanel);
        configPanel.add(patchesPanel);
        configPanel.add(patiencePanel);

        JButton trainButton = new JButton("train");
        trainButton.setActionCommand("train");
        trainButton.addActionListener(e -> controller.train());
        trainButton.setEnabled(true);
        trainButton.setAlignmentY(BOTTOM_ALIGNMENT);

        mainPanel.add(configPanel);
        mainPanel.add(trainButton);

        add(mainPanel);
    }

    private JPanel createParamPanel(String name, int value) {
        JPanel epochsPanel = new JPanel();

        epochsPanel.add(new JLabel(name + ": "));
        epochsPanel.add(new JTextField(String.valueOf(value), 4));

        return epochsPanel;
    }

    public JSONObject getTrainParams() {
        JSONObject params = new JSONObject();

        JTextField nchannels = (JTextField) nchannelsPanel.getAccessibleContext().getAccessibleChild(1);
        JTextField epochs = (JTextField) epochsPanel.getAccessibleContext().getAccessibleChild(1);
        JTextField patches = (JTextField) patchesPanel.getAccessibleContext().getAccessibleChild(1);
        JTextField patience = (JTextField) patiencePanel.getAccessibleContext().getAccessibleChild(1);

        params.put("num_channels", Integer.parseInt(nchannels.getText()));
        params.put("max_epochs", Integer.parseInt(epochs.getText()));
        params.put("npatches_epoch", Integer.parseInt(patches.getText()));
        params.put("early_stop_patience", Integer.parseInt(patience.getText()));

        return params;
    }
}
