# Drawing Tools in Java

Sources (PDF page numbers include title slides):
- `02) COT4521, 5520 slides - drawing tools in Java.pdf`
- `05) COT4521, 5520 slides - proximity.pdf`
- `08) COT4521, 5520 slides - geometric search.pdf`

## Coordinate system and drawing calls

Java component coordinates start at the top-left: `(0,0)`. Positive x moves right and positive y moves **down**. A mathematical point `(x,y)` can be displayed using a chosen screen origin `(ox,oy)` and scale `s` as `(ox+s*x, oy-s*y)`. This conversion explains why mathematical counterclockwise turns appear reversed when raw screen coordinates are used (02, p. 3).

`g.drawLine(x1,y1,x2,y2)` uses endpoints. In contrast, `drawOval(leftX,topY,width,height)` and `fillOval(...)` use a bounding rectangle, **not** a center and radius. To draw a circle centered at `(cx,cy)` with radius `r`, use `(cx-r,cy-r,2r,2r)`. For center `(80,100)` and radius 15, the bounding box is `(65,85,30,30)` (02, pp. 5–14).

`setColor` and `setFont` change the state used by subsequent drawing calls. `draw...` draws an outline; `fill...` paints the interior. `drawString(text,x,y)` places the text relative to its baseline, so its y parameter is not the top of a text box.

Drawing order matters: later drawing can cover earlier drawing. Draw connecting edges before filled vertex circles when you want the circles to hide the parts of the lines inside them. Filling after drawing an outline can cover some outline pixels; a simple pattern is fill first, outline second.

## Application, frame, panel

The lecture framework has three roles (02, pp. 7–10, 15–32, 39–48):

| Role | Responsibility |
| --- | --- |
| Application / `main` | Read or create input, construct the frame |
| `FrameDisplay extends JFrame` | Configure the window and contain the panel |
| `GraphDisplay extends JPanel` | Store drawable state and render it |

The examples override `paint(Graphics g)` and call `super.paint(g)` before custom drawing. Follow that convention when reading exam code. As a supplemental Swing practice, custom panel drawing is usually placed in `paintComponent(Graphics g)` with `super.paintComponent(g)`; this is a convention clarification, not a replacement of the professor's examples.

**Do not read console input inside painting.** The system may call the paint method repeatedly after exposure, resizing, or repaint requests. Painting should render already-stored state. Blocking for input there can freeze drawing and make redraws depend on unexpected user input.

The update flow is: change the object's coordinates or other model state, then call `repaint()`. That requests a future redraw; it is not a guarantee of synchronous painting at the call site. Calling `repaint()` without changing the model will draw the same geometry again.

## Window and panel sizes

A frame includes window decorations such as borders and the title bar. A frame size of 400 by 400 does not promise a drawable panel of 400 by 400. Give the panel a preferred size and call `pack()` on the frame so that the frame fits its contents (02, pp. 40–48).

For a 400-pixel-wide panel, valid pixel x coordinates are 0 through 399. A circle may extend outside the panel even if its center is inside. With radius 10, a center at x=395 crosses the right boundary.

## Animation and events

The animation examples repeatedly update location and request a repaint (02, pp. 33–38). A positive change to both screen coordinates moves a shape down and right. As supplemental implementation advice, avoid sleeping on Swing's event dispatch thread; a Swing timer is a common way to schedule animation without blocking events.

Mouse interaction follows three steps: implement the relevant listener interface, register the listener on the component, and respond to event coordinates by updating state and repainting. `MouseListener` covers clicks, presses, releases, entry, and exit. `MouseMotionListener` covers moving and dragging. Implementing methods without registering the listener is insufficient (05, pp. 21–36; 08, pp. 34–38).

For dragging, store the offset between the pointer and the selected object at press time. During each drag, compute the new location from the pointer and that stored offset. Otherwise the object can jump so its reference point snaps to the pointer. This applies the rectangle-drag exercise in the point-location lecture.

## Worked code reading

```java
// Circle: center (60, 90), radius 20
int x = 60 - 20, y = 90 - 20;
g.setColor(Color.ORANGE);
g.fillOval(x, y, 40, 40);
g.setColor(Color.BLACK);
g.drawOval(x, y, 40, 40);
```

The result is a filled orange disk with a black outline. Its top-left bounding-box corner is `(40,70)`. Updating its screen y coordinate by +25 moves it down 25 pixels.

## Recall checks

1. Why should geometry remain stored outside painting? **The framework can redraw at any time.**
2. Why can a correct mathematical left-turn test look clockwise on screen? **The screen y-axis is inverted.**
3. Why call `pack()` after setting the panel's preferred size? **It sizes the frame around the desired content size.**
