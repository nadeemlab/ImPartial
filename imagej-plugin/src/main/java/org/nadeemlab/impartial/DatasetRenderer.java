package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;

public class DatasetRenderer extends JLabel implements ListCellRenderer<Sample> {

    private final ImageIcon waitIcon = new ImageIcon(getClass().getClassLoader().getResource("clock.png"));
    public DatasetRenderer() {
        setOpaque(true);
        setHorizontalAlignment(LEFT);
        setVerticalAlignment(CENTER);
        setHorizontalTextPosition(RIGHT);
    }

    @Override
    public Component getListCellRendererComponent(JList<? extends Sample> list, Sample value, int index, boolean isSelected, boolean cellHasFocus) {
        if (isSelected) {
            setBackground(list.getSelectionBackground());
            setForeground(list.getSelectionForeground());
        } else {
            setBackground(list.getBackground());
            setForeground(list.getForeground());
        }

        setText(value.getName());
        if (value.getStatus().equals("running"))
            setIcon(waitIcon);
        else
            setIcon(null);

        return this;
    }
}
