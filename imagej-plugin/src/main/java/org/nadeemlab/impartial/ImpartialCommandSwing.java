package org.nadeemlab.impartial;

import org.scijava.Context;
import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import javax.swing.*;

@Plugin(type = Command.class, headless = true,
	menuPath = "Plugins>ImPartial")
public class ImpartialCommandSwing implements Command {
	@Parameter
	private Context context;

	private void createAndShowGUI() {
		ImpartialController controller = new ImpartialController(context);
	}

	/**
	 * show a dialog and give the dialog access to required IJ2 Services
	 */
	@Override
	public void run() {
		SwingUtilities.invokeLater(this::createAndShowGUI);
	}
}
