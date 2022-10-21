package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ModelPanel extends JPanel {
    ModelPanel(ImpartialDialog controller) {
        setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createTitledBorder("model"),
            BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JButton trainButton = new JButton("train");
        trainButton.setActionCommand("train");
        trainButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.train();
            }
        });
        trainButton.setEnabled(true);

        JButton inferButton = new JButton("infer");
        inferButton.setActionCommand("infer");
        inferButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                controller.infer();
            }
        });
        inferButton.setEnabled(true);

        add(trainButton);
        add(inferButton);
    }


}
