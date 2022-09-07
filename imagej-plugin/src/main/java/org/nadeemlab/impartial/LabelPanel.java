package org.nadeemlab.impartial;

import javax.swing.*;

public class LabelPanel extends JPanel {
    private final ImpartialController controller;

    LabelPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("label"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        add(createLabelPanel());
    }

    private JPanel createLabelPanel() {
        JPanel buttonsPanel = new JPanel();
        buttonsPanel.setLayout(new BoxLayout(buttonsPanel, BoxLayout.LINE_AXIS));

        JButton loadLabelButton = new JButton("open");
        loadLabelButton.addActionListener(e -> controller.loadLabel());

        JButton submitLabelButton = new JButton("submit");
        submitLabelButton.addActionListener(e -> controller.submitLabel());

        buttonsPanel.add(loadLabelButton);
        buttonsPanel.add(submitLabelButton);

        return buttonsPanel;
    }
}
