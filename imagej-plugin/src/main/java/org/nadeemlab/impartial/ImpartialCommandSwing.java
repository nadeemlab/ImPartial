package org.nadeemlab.impartial;

import org.scijava.Context;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import javax.swing.*;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>ImPartial")
public class ImpartialCommandSwing implements Command {
    @Parameter
    private Context context;

    private void createAndShowGUI() {
        ImpartialController controller = new ImpartialController(context);
    }

    @Override
    public void run() {
        SwingUtilities.invokeLater(this::createAndShowGUI);
    }
}
