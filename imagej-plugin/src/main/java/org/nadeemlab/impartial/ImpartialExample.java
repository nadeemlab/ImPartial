package org.nadeemlab.impartial;

import javax.swing.*;
import net.imagej.ImageJ;

public class ImpartialExample {

	public static void main(final String[] args) {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		try {
			ImpartialController controller = new ImpartialController(ij.context());
		}
		catch (final Exception e) {
			e.printStackTrace();
		}
	}

}
